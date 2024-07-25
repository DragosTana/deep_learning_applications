import torch
import torch.nn as nn
import torch.optim as optim

class JARN:
    def __init__(self, 
                 classifier, 
                 discriminator, 
                 adaptor, 
                 lambda_adv, 
                 device, 
                 optimizer=optim.Adam, 
                 learning_rate=0.001,
                 epsilon=8/255): 
        
        self.classifier = classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.adaptor = adaptor.to(device)
        self.lambda_adv = lambda_adv
        self.device = device
        self.epsilon = epsilon
        self.iteration_count = 0  # To keep track of iterations
        self.disc_update_interval = 0

        self.cls_optimizer = optimizer(self.classifier.parameters(), lr=learning_rate)
        self.disc_optimizer = optimizer(self.discriminator.parameters(), lr=learning_rate)
        self.apt_optimizer = optimizer(self.adaptor.parameters(), lr=learning_rate)

        self.criterion = nn.CrossEntropyLoss()

    def compute_jacobian(self, x, y):
        x.requires_grad_(True)
        y_pred = self.classifier(x)
        loss = self.criterion(y_pred, y)
        grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        return grad

    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        
        # Add uniform noise to input
        x = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        
        # Train classifier
        self.cls_optimizer.zero_grad()
        y_pred = self.classifier(x)
        cls_loss = self.criterion(y_pred, y)
        
        jacobian = self.compute_jacobian(x, y)
        adapted_jacobian = self.adaptor(jacobian)
        
        disc_real = self.discriminator(x)
        disc_fake = self.discriminator(adapted_jacobian)
        
        adv_loss = torch.log(disc_real).mean() + torch.log(1 - disc_fake).mean()
        total_loss = cls_loss + self.lambda_adv * adv_loss
        total_loss.backward()
        self.cls_optimizer.step()
        
        # Train adaptor
        self.apt_optimizer.zero_grad()
        adapted_jacobian = self.adaptor(jacobian.detach())
        disc_fake = self.discriminator(adapted_jacobian)
        apt_loss = -torch.log(disc_fake).mean()
        apt_loss.backward()
        self.apt_optimizer.step()
        
        # Train discriminator only every disc_update_interval iterations
        if self.iteration_count % self.disc_update_interval == 0:
            self.disc_optimizer.zero_grad()
            disc_real = self.discriminator(x)
            disc_fake = self.discriminator(adapted_jacobian.detach())
            disc_loss = -torch.log(disc_real).mean() - torch.log(1 - disc_fake).mean()
            disc_loss.backward()
            self.disc_optimizer.step()
        
        self.iteration_count += 1  # Increment the iteration count
        
        return total_loss.item()

    def train(self, train_loader, num_epochs, disc_update_interval=20):
        self.disc_update_interval = disc_update_interval
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                loss = self.train_step(x, y)
                epoch_loss += loss
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss}')
            
            print(f'Epoch {epoch}, Average Loss: {epoch_loss / len(train_loader)}')


    def evaluate(self, test_loader):
        self.classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.classifier(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy}%')
        self.classifier.train()
        return accuracy
