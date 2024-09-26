#importações comuns
import pygame 
import random
import math

#importações da machiner learn
import torch
import torch.nn as nn
import torch.optim as optim


#Inicia o pygame
pygame.init();

#Criar o display
telaLargura = 600;
telaAltura = 600;
tela = pygame.display.set_mode((telaLargura, telaAltura));
pygame.display.set_caption("Lizard and the lake");

#Definido as cores
SAND = (252, 233, 179);
GREEN = (111, 237, 38);
BLUE = (100, 183, 240);
BLACK = (0, 0, 0)

#Parâmetros da rede neural
input_size = 2  # Distância e sede
output_size = 4 # Ações: cima, baixo, esquerda, direita

epsilon = 0.2  # Chance de explorar (10% no início)
taxa = 0.001   # Taxa de aprendizado


#Função padrão
def gerar_posicao_distante_lago(x1, y1, tamanho):
        distancia_minima = 100;
        while True:
            x = random.randint(0, telaLargura - tamanho);
            y = random.randint(0, telaAltura - tamanho);
            distancia = math.sqrt((x - x1)**2 + (y - y1)**2);
            if distancia >= distancia_minima:
                return [x, y, distancia];
            
def calcular_recompensa(personagem, lago):
        #distancia atual
        distancia = math.sqrt((personagem.x - lago.x)**2 + (personagem.y - lago.y)**2);

        #Recompença com base na comparação das distancias
        if distancia < personagem.distancia:
            recompensa = 10;
        else:
            recompensa = -10;
            
    
        #Recompensa com base na sede
        if personagem.sede > 50:
            recompesa = 1;
        elif personagem.sede < 50 and personagem.sede > 20:
            recompensa = 0;
        else:
            recompensa = -2;
            
        #penalidade por tocar as bordas
        if personagem.y <= 0 or personagem.y >= telaAltura - personagem.tamanho or personagem.x <= 0 or personagem.x >= telaLargura - personagem.tamanho:
            recompensa = -50;   
        

        #Recompeça com base na colisão
        if personagem.colidir == True:
            recompensa = 100;

        #Recompença com base na vida
        if personagem.vida == False:
            recompensa = -100;

        return recompensa
            
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
            
#Class para personagem
class Personagem:
    
    def __init__(self, x, y, velocidade = 2.5, tamanho = 20):
        self.x = x // 2;
        self.y = y // 2;
        self.distancia = 500
        self.velocidade = velocidade;
        self.sede = 100;
        self.tamanho = tamanho;
        self.vida = True;
        self.colidir = False;
        

    def  mover(self, direcao):
        if self.vida == True:
            if direcao == "cima" and self.y > 0:
                self.y -= self.velocidade;
            if direcao == "baixo" and self.y < telaAltura - self.tamanho:
                self.y += self.velocidade;
            if direcao == "esquerda" and self.x > 0:
                self.x -= self.velocidade;
            if direcao == "direita" and self.x < telaLargura - self.tamanho:
                self.x += self.velocidade;

            self.sede -= 0.2;

    def renascer(self, x, y, lago):
        self.vida = True;
        self.x = x // 2;
        self.y = y // 2;
        self.sede = 100;
        self.verificar_distancia(lago);

    def desenhar(self, tela):
        if self.vida == True:
            pygame.draw.rect(tela, GREEN, (self.x, self.y, self.tamanho, self.tamanho));

    def verificar_sede(self):
        if self.sede <= 0:
            self.vida = False;
        else:
            self.vida = True;

    def verificar_distancia(self, lago):
        self.distancia = math.sqrt((self.x - lago.x)**2 + (self.y - lago.y)**2);
        print(self.distancia)

    def verificar_colisao(self, lago):
        if pygame.Rect(self.x, self.y, self.tamanho, self.tamanho).colliderect(lago.x, lago.y, lago.tamanho, lago.tamanho):
            self.sede += 100;
            lago.reposicionar();
            self.colidir = True;
        else:
            self.colidir = False;

#---------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------- 

#Classe para Lago
class Lago:
    def __init__(self, personagem):
        self.personagem = personagem;
        self.tamanho = 50;
        pos = gerar_posicao_distante_lago(self.personagem.x, self.personagem.y, self.tamanho);
        self.x = pos[0];
        self.y = pos[1];
        
    def desenhar(self, tela):
        pygame.draw.rect(tela, BLUE, (self.x, self.y, self.tamanho, self.tamanho));

    def reposicionar(self):
        pos = gerar_posicao_distante_lago(self.personagem.x, self.personagem.y, self.tamanho);
        self.x = pos[0];
        self.y = pos[1];

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
        
#CLasse para a rede neural
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__();
        
        
        #camadas da rede neural
        self.fc1 = nn.Linear(input_size, 128);  # Camada oculta com 128 neurônios
        self.fc2 = nn.Linear(128, 64);          # Camada oculta com 64 neurônios
        self.fc3 = nn.Linear(64, output_size);  # Camada de saída (representa as ações)

    def forward(self, x):
        x = torch.relu(self.fc1(x));  # Função de ativação ReLU na primeira camada
        x = torch.relu(self.fc2(x));  # Função de ativação ReLU na segunda camada
        x = self.fc3(x);              # Saída (as ações)

        return x;

    def escolher_acao(self, estado):
        if random.random() < epsilon:
            acao = random.randint(0, 3);
        else:
            # Exploração: usar a rede neural para escolher a melhor ação
            estado_tensor = torch.FloatTensor(estado);
            with torch.no_grad():
                acao_valores = net(estado_tensor);         # Passa o estado pela rede neural
                acao = torch.argmax(acao_valores).item();  # Ação com maior valor (a melhor ação)
                
        return acao
    
    def treinar(self, estado, acao, recompensa, proximo_estado, loss, optimizer):
        estado_tensor = torch.FloatTensor(estado)
        proximo_estado_tensor = torch.FloatTensor(proximo_estado)
        acao_tensor = torch.LongTensor([acao])
        recompensa_tensor = torch.FloatTensor([recompensa])

        # Predição do valor atual
        predicao_atual = self(estado_tensor)[acao_tensor]

        # Predição do valor futuro
        with torch.no_grad():
            valor_futuro = self(proximo_estado_tensor).max()

        # Valor alvo
        valor_alvo = recompensa_tensor + 0.99 * valor_futuro

        # Calcula a perda
        perda = loss(predicao_atual, valor_alvo)

        # Atualiza os pesos
        optimizer.zero_grad()
        perda.backward()
        optimizer.step()
    
            

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
        
#Classe para o mapa
class Mapa:
    def __init__(self, net, loss, optimizer):
        self.personagem = Personagem(telaLargura, telaAltura);
        self.lago = Lago(self.personagem);
        self.relogio = pygame.time.Clock();
        self.rede = net;
            
    def PLAYER(self):
        play = True;
        while play:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    play = False;

            estado = [self.personagem.distancia, self.personagem.sede];

            acao = self.rede.escolher_acao(estado);
            
            if acao == 0:
                self.personagem.mover("cima");
            elif acao == 1:
                self.personagem.mover("baixo");
            elif acao == 2:
                self.personagem.mover("esquerda");
            elif acao == 3:
                self.personagem.mover("direita");
                
            
            self.personagem.verificar_colisao(self.lago);
            self.personagem.verificar_sede();

            recompensa = calcular_recompensa(self.personagem, self.lago);

            self.personagem.verificar_distancia(self.lago);
            
            # Verificar se o jogo acabou
            if(self.personagem.vida == False):
                self.personagem.renascer(telaLargura, telaAltura, self.lago)

            # Obter o próximo estado
            proximo_estado = [self.personagem.sede, self.personagem.distancia]

            # Treinar a rede neural
            self.rede.treinar(estado, acao, recompensa, proximo_estado, loss, optimizer)

            
            tela.fill(SAND);

            #desenhar na tela
            self.personagem.desenhar(tela);
            self.lago.desenhar(tela);
            
            pygame.display.update();

            # Controlar a taxa de atualização
            self.relogio.tick(30);

        pygame.quit();
#---------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------- 
    
if __name__ == "__main__":
    net = NeuralNet(input_size, output_size);
    loss = nn.MSELoss();
    optimizer = optim.Adam(net.parameters(), lr=taxa);
    ambiente = Mapa(net, loss, optimizer);
    ambiente.PLAYER();
