import numpy
import random
import connection as cn

ACTIONS = ['left','right','jump']
DIRECTIONS = ['north','east','south','west']
Q_TABLE = None


REWARD_MAP =    [ 
                [-100, -3, -4, -5, -6, -100, -100, -100],
                [-100, -2, -100, -6, -7, -8, -100, -100],
                [300, -1, -100, -100, -8, -9, -10, -100],
                [300, -1, -100, -100, -100, -10, -11, -100],
                [-100, -2, -100, -100, -100, -100, -12, -100],
                [-100, -3, -4, -5, -7, -12, -13, -100],
                [-100, -100, -100, -100, -100, -100, -14, -100],
                [-100, -100, -100, -100, -100, -100, -100, -100]
                ]

PLATFORMS = [   
                [-1, 11, 10, 9, 8, -1, -1, -1],
                [-1, 12, -1, 16, 7, 6, -1, -1],
                [-1, 13, -1, -1, 15, 5, 4, -1],
                [-1, 23, -1, -1, -1, 14, 3, -1],
                [-1, 22, -1, -1, -1, -1, 2, -1],
                [-1, 21, 20, 19, 18, 17, 1, -1],
                [-1, -1, -1, -1, -1, -1, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1]
            ]
X = 6
Y = 6
DIR = 0
FAIL_CHANCE = 10

def fake_state_reward(action):
    global X
    global Y
    global DIR

    match action:
        case 'left':
            DIR = (DIR - 1) % 4
        case 'right':
            DIR = (DIR + 1) % 4
        case 'jump':
            r = random.randint(1, 100)

            if r > FAIL_CHANCE:
                match DIR:
                    case 0:
                        Y -= 1
                    case 1:
                        X += 1
                    case 2:
                        Y += 1
                    case 3:
                        X -= 1
            else:
                match DIR:
                    case 1:
                        Y -= 1
                    case 2:
                        X += 1
                    case 3:
                        Y += 1
                    case 0:
                        X -= 1
    
    reward = REWARD_MAP[Y][X]

    if reward == -100 or reward == 300:
        X=6
        Y=6
    
    bit_state = bin((PLATFORMS[Y][X] << 2) + DIR)

    return (bit_state, reward)

def save_table(location, table):
    numpy.savetxt(location, table)

def load_table(location):
    try:
        with open(location) as file:
            table = numpy.loadtxt(file)
    except FileNotFoundError:
        print("Arquivo não encontrado, criando uma tabela vazia")
        table = load_empty_table()

    print("table: ", table[0])
    return table

def load_empty_table():
    row = [0.0] * len(ACTIONS)
    table = [row] * 96
    table = numpy.array(table)
    return table

#aplcia a função de Q_learning e atualiza na tabela
def update_table(reward, prev_s, prev_a, curr_s, learning_rate = 0.01, discount = 0.9):
    bellman = reward + discount * max(Q_TABLE[curr_s])

    Q_TABLE[prev_s][prev_a] += learning_rate * (bellman - Q_TABLE[prev_s][prev_a])
    return Q_TABLE[prev_s][prev_a]

def extract_state(bit_state):
    plat_mask = 0b1111100
    direction_mask = 0b0000011

    int_state = int(bit_state, 2) #converte de binário pra decimal
    #aplica uma mascara de bits e faz um shift right pra que os bits fiquem no local correto
    platform = (int_state & plat_mask) >> 2
    direction = (int_state & direction_mask)

    #print(f"Bits: {bit_state}  \nPlataforma:  {platform} \nDireção: {DIRECTIONS[direction]}\n")

    #acha o estado entre 0-95 a partir da direção e plataforma
    state = platform * 4 + direction
    return state
    
def navigate(socket, times, start):
    #reseta o estado atual
    current_state = start * 4
    total_reward = 0

    for i in range(times):
        #busca na tabela a melhor ação pra esse estado
        current_action = numpy.where(Q_TABLE[current_state] == max(Q_TABLE[current_state]))[0][0]

        #recebe a recompensa e o estado novos a partir da ação escolhida
        bit_state, reward = cn.get_state_reward(socket, ACTIONS[current_action])
        new_state = extract_state(bit_state)

        total_reward += reward
        current_state = new_state

        print('Ação feita: ' + ACTIONS[current_action])
        print(f'Novo estado: {new_state}')
        print(f'Recompensa média: {round(total_reward/(i+1), 3)} \n')


def explore(socket, times, start):
    #reseta o estado atual
    current_state = start * 4 # *4, pois cada plataforma tem 4 estados possíveis e sempre respawna virado pro norte

    for i in range(times):
        print('Passo: ' , i)

        #busca na tabela a melhor ação para o estado atual
        best_action = numpy.where(Q_TABLE[current_state] == max(Q_TABLE[current_state]))[0][0]
        rand_action = random.randint(0, 2)

        #escolhe entre uma ação aleatória ou a melhor ação
        if i % 5 > 1:
            current_action = best_action
        else:
            current_action = rand_action

        #recebe a recompensa e o estado novos a partir da ação escolhida
        bit_state, reward  = cn.get_state_reward(socket, ACTIONS[current_action])

        new_state = extract_state(bit_state)

        update_table(reward, current_state, current_action, new_state)

        current_state = new_state

def fake_explore(times):
    global X
    global Y
    global DIR
    X=6
    Y=6
    DIR=0

    #reseta o estado atual
    current_state = 0

    for i in range(times):
        #busca na tabela a melhor ação para o estado atual
        best_action = numpy.where(Q_TABLE[current_state] == max(Q_TABLE[current_state]))[0][0]
        rand_action = random.randint(0, 2)

        #escolhe entre uma ação aleatória ou a melhor ação
        if i % 5 > 1:
            current_action = best_action
        else:
            current_action = rand_action

        #recebe a recompensa e o estado novos a partir da ação escolhida

        bit_state, reward  = fake_state_reward(ACTIONS[current_action])

        new_state = extract_state(bit_state)

        update_table(reward, current_state, current_action, new_state)

        current_state = new_state

def main():
    global Q_TABLE

    socket = cn.connect(2037)

    if(socket != 0):
        while True:
            command = input('Envie um comando\n')
            match command:
                case 'save':
                    save_table('resultado.txt', Q_TABLE)
                    print('Tabela salva')

                case 'empty':
                    Q_TABLE = load_empty_table()
                    print('Tabela vazia carregada')

                case 'load':
                    Q_TABLE = load_table('resultado.txt')
                    print('Tabela carregada')

                case 'explore':
                    if Q_TABLE is None:
                        print('Carregue uma tabela antes')
                    else:
                        try:
                            times = int(input('Quantas vezes ele deve explorar?\n'))
                            start = int(input('Qual a plataforma inicial?\n'))
                        except ValueError:
                            print('Valor inválido')
                        else:
                            explore(socket, times, start)
                            print('Exploração terminada')

                case 'fake':
                    if Q_TABLE is None:
                        print('Carregue uma tabela antes')
                    else:
                        try:
                            times = int(input('Quantas vezes ele deve explorar?\n'))
                        except ValueError:
                            print('Valor inválido')
                        else:
                            fake_explore(times)
                            print('Exploração terminada')

                case 'navigate':
                    if Q_TABLE is None:
                        print('Carregue uma tabela antes')
                    else:
                        try:
                            times = int(input('Quantas vezes ele deve percorrer?\n'))
                            start = int(input('Qual a plataforma inicial?\n'))
                        except ValueError:
                            print('Valor inválido')
                        else:
                            navigate(socket, times, start)

                case 'exit':
                    socket.close()
                    break

if __name__ == "__main__":
    main()
