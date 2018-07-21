
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[57]:




# In[2]:


R = 250
word = np.zeros([R, R, R])


# In[12]:


def mlen(d):
    return np.abs(d).sum()
def clen(d):
    return np.abs(d).max()
def adjacent(d):
    return mlen(d) == 1

def is_linear(d):
    return ((d[0] != 0 and d[1] == 0 and d[2] == 0) or 
            (d[0] == 0 and d[1] != 0 and d[2] == 0) or 
            (d[0] == 0 and d[1] == 0 and d[2] != 0))

def short_linear(d):
    return mlen(d) <= 5 
def long_linear(d):
    return mlen(d) <= 15 

def near(d):
    return mlen(d) <= 2 and clen(d) == 1

# Regions
def in_region(c1, c2, c):
    return (min(c1[0], c2[0]) <= c[0] <= max(c1[0], c2[0]) and 
            min(c1[1], c2[1]) <= c[1] <= max(c1[1], c2[1]) and 
            min(c1[2], c2[2]) <= c[2] <= max(c1[2], c2[2]))

def dim_region(c1, c2):
    return (c1[0] == c2[0]) + (c1[1] == c2[1]) + (c1[2] == c2[2])

def bytes_from_file(filename, chunksize=8192):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break
def bits(n):
    while n:
        b = n & (~n+1)
        yield b
        n ^= b


# In[715]:



def ld(d, off, pad):
    if d[0] != 0:
        a = "01"
        val = d[0]
    if d[1] != 0:
        a = "10"
        val = d[1]
    if d[2] != 0:
        a = "11"
        val = d[2]
    i = val + off
    b = bin(i)[2:]
    return a, "0"*(pad - len(b)) + b
        
def nd(d):
    b = bin((d[0] + 1 )* 9  + (d[1] + 1) * 3 +   d[2] + 1)[2:]
    return "0"*(5 - len(b)) + b
    
class Command:
    def __init__(self, typ, d1 = None, d2=None):
        self.typ = typ
        self.d1 = d1
        self.d2 = d2
        
    def __repr__(self):
        return "[%s %s %s]"%(self.typ, self.d1 if self.d1 is not None else "", self.d2 if self.d2 is not None else "")

class Halt(Command):
    def __init__(self):
        super().__init__("Halt")
        
    def write(self):
        return ["11111111"]

class Wait(Command):
    def __init__(self):
        super().__init__("Wait")
    
    
    def write(self):
        return ["11111110"]

class Flip(Command):
    def __init__(self):
        super().__init__("Flip")
    
    
    def write(self):
        return ["11111101"]

class SMove(Command):
    def __init__(self, d):
        super().__init__("SMove", d)
        assert(d[0] <= 15)
        assert(d[1] <= 15)
        assert(d[2] <= 15)
        
    def write(self):
        a, i = ld(self.d1, 15, 5)
        return ["00" + a + "0100",  "000" + i ]
    
class LMove(Command):
    def __init__(self, d1, d2):
        super().__init__("LMove", d1, d2)
    
    def write(self):
        a1, i2 = ld(self.d1, 5, 4)
        a2, i2 = ld(self.d2, 5, 4)
        return [a2 + a1+ "1100", i2 + i1]
    
class Fission(Command):
    def __init__(self, d1, d2):
        super().__init__("Fission", d1, d2)
    
    def write(self):
        nd1 = nd(self.d1)
        b = bin(self.d2)[2:]
        return [nd1 + "101", "0"*(8-len(b)) + b ]
    
class FusionP(Command):
    def __init__(self, d1):
        super().__init__("FusionP", d1)
    
    def write(self):
        nd1 = nd(self.d1)
        return [nd1 + "111"]

class FusionS(Command):
    def __init__(self, d1):
        super().__init__("FusionS", d1)
    
    def write(self):
        nd1 = nd(self.d1)
        return [nd1 + "110"]

class Fill(Command):
    def __init__(self, d1):
        super().__init__("Fill", d1)
    
    def write(self):
        nd1 = nd(self.d1)
        return [nd1 + "011"]


# In[102]:


def load_commands(f="traces/LA001.nbt"):
    commands = []
    useds = []
    inp = bytes_from_file(f)
    def conv(b):
        b = bin(b)[2:]
        return "0"* (8-len(b)) + b
    while True:
        try:
            b = conv(next(inp))
        except StopIteration:
            break

        def ld(a, b, off):
            val = int(b, 2) - off
            if a == "01":
                return np.array([val, 0, 0])
            if a == "10":
                return np.array([0, val, 0])
            if a == "11":
                return np.array([0, 0, val])
            assert False
            
        def nd(a):
            a = int(a, 2)
            return np.array([a // 9 - 1 , (a // 3) % 3 -1  , (a % 3) - 1 ] )
        
        used = [b]
        if b == "11111111":
            com = Halt()
            
        elif b == "11111110":
            com = Wait()
        elif b == "11111101":
            com = Flip()
        elif b[-4:] == "0100":
            c = conv(next(inp))
            used.append(c)
            d = ld(b[2:4], c[3:], 15)
            com = SMove(d)

        elif b[-4:] == "1100":
            c = conv(next(inp))
            used.append(c)
            d1 = ld(b[0:2], c[0:4], 5)
            d2 = ld(b[2:4], c[4:], 5)
            com = LMove(d1, d2)
            
        elif b[-3:] == "111":
            d = nd(b[:5])
            com = Command("FusionP", d)
        elif b[-3:] == "110":
            d = nd(b[:5])
            com = Command("FusionS", d)
        elif b[-3:] == "101":
            c = conv(next(inp))
            used.append(c)
            d1 = nd(b[:5])
            print(c)
            d2 = int(c, 2)
            com = Fission(d1, d2)
        elif b[-3:] == "011":
            d = nd(b[:5])
            com = Fill(d)
        else:
            assert False
        commands.append(com)
        useds.append(used)
    return commands, useds


# In[298]:


commands, useds = load_commands()
for c, u in zip(commands, useds):
    #print(c, u, c.write())
    if u != c.write():
        assert False
        
def write_commands(commands, f):
    t = open(f, "wb")
    for c in commands:
        for b in c.write():
            t.write(bitstring_to_bytes(b))
    t.close()


# In[118]:




# In[107]:


def bitstring_to_bytes(s):
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')


# In[108]:


bitstring_to_bytes("00000000")


# In[126]:


def run_group(state, commands):
    #for bot, command in zip(commands, state.bots):
    #    set_interference(command, state, interference)
    n = len(state.bots)
    to_run = commands[:n]
    state.update()
    for command, bot in zip(to_run, state.bots):
        update_state(command, bot, state)
    
    return commands[n:]
    
def update_state(command, bot, S):
    com = command
    #print(command)
    c = bot.pos
    if command.typ == "Halt":
        if  len(S.bots) != 1 or S.harmonics != False:
            assert False
        S.bots = []
        return
    if command.typ == "Wait":
       return

    if command.typ == "Flip":
        S.harmonics = not S.harmonics

    if command.typ == "SMove":
        assert(long_linear(com.d1))
        bot.pos = bot.pos + com.d1
        S.energy = S.energy + 2 * mlen(com.d1)

    if command.typ == "LMove":
        assert(short_linear(com.d1) and short_linear(com.d2))
        bot.pos = bot.pos + com.d1
        bot.pos = bot.pos + com.d2

        S.energy = S.energy + 2 * (mlen(com.d1) + 2 + mlen(com.d2))

    if command.typ == "Fission":
        if not bot.seeds: assert False
        b1 = bot.seeds[0]
        bot.seeds = bot.seeds[1:]
        # make a new bot
        bs = BotState(b1, bot.pos + com.d1, bot.seeds[1:com.d2])
        bot.seeds = bot.seeds[com.d2:]
        S.bots.append(bs)
        S.energy = S.energy + 24
    
    if command.typ == "Fill":
        spot = bot.pos + com.d1
        if S.matrix[spot[0], spot[1], spot[2]] == 0:
            S.matrix[spot[0], spot[1], spot[2]] = 1
            S.energy = S.energy + 12
        else:
            S.energy = S.energy + 6


# In[127]:


# model, R = load_model("models/LA001_tgt.mdl")
# commands, _ = load_commands("traces/LA001.nbt")
# state = NanoState.start(R)
# while len(commands) > 0: 
#     commands = run_group(state, commands)
# state.show()
# state.energy


# In[495]:



def order(M, R):
    pos = []
    graph = {"start" : set()}
    m2 = np.zeros([R, R, R])
    m2.fill(1000)
    for x in range(R):
        for z in range(R):
            if M[x, 0, z] == 1:
                m2[x, 0, z] = 1
                pos.append({"pos":np.array([x, 0, z]), "order":1 })
                graph["start"].add((x, 0, z))
    #for q in range(3*R):
    for x in range(R):
        for y in range(R):
            for z in range(R):
                if M[x, y, z] == 1 and m2[x, y, z] == 1000:
                    
                    for v in[[x-1, y, z], [x+1, y, z], [x, y-1, z],[x, y+1, z], 
                            [x, y, z-1], [x, y, z+1]]:
                        if M[v[0], v[1], v[2]] == 1:
                            graph.setdefault(tuple(v), set())
                            graph[tuple(v)].add( (x, y, z) )
                        #if v.min() < 1000:
                        #    m2[x,y,z] = v.min()+1
                        #    pos.append({"pos":np.array([x, y, z]), "order":v.min()+1}) 
    return graph



def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
def order_targets(graph, R, n, sp):

    targets_todo = [{} for player in range(n)]
    start = [t for t in graph["start"]]
    bot_pos = []
    for player in range(n):
        print(n)
        k = int(len(start) / n)
        targets_todo[player][0] = set(start[player * k: (player+1) * k 
                                            if player != n -1 else len(start)])
        bot_pos.append(sp[player])
    targets = {}
        #local_targets = [p for p in pos if p["order"] == level]
    done = set()
    player_done = [False for _ in range(n)]
    while True:
        for player in range(n):
            targets.setdefault(player, [])
            m = 1e10
            for k in sorted(targets_todo[player].keys()):
                if not targets_todo[player][k]: 
                    continue
                targ = None
                bp = bot_pos[player]
                r = [(bp[0] + 1, bp[1], bp[2]),
                     (bp[0] - 1, bp[1], bp[2]),
                     (bp[0], bp[1], bp[2] + 1),
                     (bp[0], bp[1], bp[2] - 1)]
                for t in r:
                    if t in targets_todo[player][k]:
                        targ = t
                        m = 1
                        break
                if m == 1e10:
                    for t in targets_todo[player][k]: #list(graph.get(bot_pos, [])) + list(targets_todo):
                    #if t not in targets_todo: continue
                        l = dist(bp, t) #+ t[1] * 20
                        if l < m:
                            targ = t
                            m = l
                            if l == 1: 
                                break            
                targets[player].append(np.array(targ))
                tt = (targ[0], targ[1], targ[2] )
                for playerp in range(n):
                    if targ[1] in targets_todo[playerp] and tt in targets_todo[playerp][targ[1]]:
                        targets_todo[playerp][targ[1]].remove(tt)
                done.add(tt)
                bot_pos[player] = targ
                for t in graph.get(targ, []):
                    if t in done: 
                        continue
                    targets_todo[player].setdefault(t[1], set())
                    targets_todo[player][t[1]].add(t)
                break
            if m == 1e10:
                player_done[player] = True
        if all(player_done):
            break
        #print(len(targets), len(targets_todo))
    #for player in range(n):
    #    targets[player].append(np.array(sp[player]))
    
    return targets


# In[1010]:


def stuck(d):
    return mlen(d) == 3 and clen(d) == 1

def make_commands(targets, R, start_pos):

        
    
    n = len(targets)
    bot_pos = {}
    for player in range(n):
        bot_pos[player] = np.array(start_pos[player])
    done = np.zeros([R+1, R+1, R+1], dtype=bool)
    inter = np.zeros([R+1, R+1, R+1], dtype=bool)
    commands = []
    def dif(x=0, y=0, z=0):
        return np.array([x, y, z])
    
    t = {}
    ti = {}
    for p in range(n):
        t[p] = targets[p][0]
        ti[p] = 0
    player_broke = [0 for p in range(n)]
    player_done = [False for p in range(n)]
    end_game = False
    bots_left = list(range(n))
    counter = 0
    rank = [0 for _ in range(n)]
    while True:
        counter += 1
        if end_game == "over":
            break
        if all(player_done):
            for p in range(n):
                player_done[p] = False
                t[p] = (0, 0, 0)
                end_game = True
                print("endgame")
            #break
        #print(player_done)
        inter.fill(0)
        
        start_com = len(commands)
        for p in bots_left:
            waiting = False
            
            if p == 0 and end_game is True:
                acted = False
                for p2 in bots_left:
                    if p2 == 0: continue
                    d = bot_pos[0] - bot_pos[p2]
                    #print(d, p2)
                    if near(d):
                        #print(bots_left)
                        commands.append(FusionP(-d))
                        for p3 in bots_left:
                            if p3 == 0: continue
                            if p3 != p2:
                                commands.append(Wait())
                            else:
                                commands.append(FusionS(d))
                        acted = True
                        bots_left = [b for b in bots_left if b != p2]
                        #print(bots_left)
                        if bots_left == [0]: end_game = "finish"
                if acted:
                    break
            if player_done[p] or any([t[p][1] > t[p2][1] for p2 in bots_left 
                                      if not player_done[p2]]):
                #bp = bot_pos[p]
                waiting = True
                #inter[bp[0], bp[1], bp[2]] = 1
                
            moved = False
            bp = bot_pos[p]
            d = t[p] - bp
            tp = t[p]
            #f = t[p] - future
            # ldone = done + inter

            extra = {}
            for p2 in bots_left:
                if p2 != p:
                    if t[p2][0] != bp[0] or t[p2][1] != bp[1] or t[p2][2] != bp[2]:
                        extra[t[p2][0], t[p2][1], t[p2][2]] = 1
                    extra[bot_pos[p2][0], bot_pos[p2][1], bot_pos[p2][2]] = 1
            def ldone(a, b, c):
                return done[a, b, c] or inter[a, b, c] or (a, b, c) in extra
            
            if player_done[p]:
                d2 = dif(x=max(-15, -bp[0]))                    
                s = -1

                for x in range(0, abs(d2[0]) + 1):
                    #print(bp[0] + s*x, bp[1], bp[2])
                    if ldone(bp[0] + s*x, bp[1], bp[2]) or (bp[0] + s*x) < 0:
                        break
                    d2[0] = s * x 
                #assert d2[0] != 0
                if d2[0] != 0:
                    commands.append(SMove(d2))
                    moved = True
                if not moved:
                    d2 = dif(z=max(-15, -bp[2] ) )                    
                    s = 1 if d2[2] > 0 else -1
                    
                    for x in range(0, abs(d2[2]) + 1):
                        if ldone(bp[0], bp[1], bp[2] + s*x) == 1 or (bp[2] + s*x) < 0:
                            break
                        d2[2] = s * x 
                    #assert d2[0] != 0
                    if d2[2] != 0:
                        #print("z move")
                        commands.append(SMove(d2))
                        moved = True
                    
            if not player_broke[p] and  not waiting and (abs(d[0]) > 1 or stuck(d)) and not moved:
                d2 = dif(x=max(min(d[0], 15), -15))                    
                s = 1 if d2[0] > 0 else -1

                for x in range(0, abs(d2[0]) + 1):
                    if ldone(bp[0] + s*x, bp[1], bp[2]):
                        break
                    d2[0] = s * x 
                #assert d2[0] != 0
                if d2[0] != 0:
                    commands.append(SMove(d2))
                    moved = True

            if not player_broke[p] and  not waiting and (abs(d[2]) > 1 or stuck(d)) and not moved:
                d2 = dif(z=max(min(d[2], 15), -15)) 
                s = 1 if d2[2] > 0 else -1

                for z in range(0, abs(d2[2]) + 1):
                    if ldone(bp[0], bp[1], bp[2] + s*z):
                        break
                    d2[2] = s * z 
                if d2[2] != 0:
                    moved = True
                    commands.append(SMove(d2))

            if not player_broke[p] and not waiting and (abs(d[1]) > 1 or stuck(d)) and not moved:
                d2 = dif(y=max(min(d[1], 15), -15))
                s = 1 if d2[1] > 0 else -1
                for x in range(0, abs(d2[1]) + 1):
                    #print(done[bp[0], s*x + bp[1], bp[2]], 
                    #      inter[bp[0], s*x + bp[1], bp[2]],
                    #     [bp[0], s*x + bp[1], bp[2]])
                    if ldone(bp[0], s*x + bp[1], bp[2]):
                        break
                    d2[1] = s * x 
                #print(d2)
                if d2[1] != 0:
                    commands.append(SMove(d2))
                    moved = True



            if not player_broke[p] and not waiting and near(d) and not moved and ldone(tp[0], tp[1], tp[2]) == 0:
                if tp[0] == 0 and (tp[1] == 0 or tp[1] == 1) and tp[2] == 0:
                    if d[0] !=0: commands.append(SMove(dif(x=d[0])))
                    if d[1] !=0: commands.append(SMove(dif(y=d[1])))
                    if d[2] !=0: commands.append(SMove(dif(z=d[2])))
                    commands.append(Halt())
                    end_game = "over"
                else:
                    commands.append(Fill(d))

                done[tp[0], tp[1], tp[2]] = 1
                moved = True
                ti[p] += 1
                if ti[p] < len(targets[p]):
                    t[p] = targets[p][ti[p]]
                else:
                    player_done[p] = True
                d2 = dif()
                #d2 = dif(y = max(min(d[1]+1, 15), -15))
                #commands.append(SMove(d2))
                #moved = True
            
            if not moved:
                if d[1] >= 0: 
                    d2 = dif(y = 1)
                    if  bp[1] + 1 < R and ldone(bp[0], bp[1] + 1, bp[2]) == 0:
                        commands.append(SMove(d2))
                        moved = True
                #if not moved:
                #    d2 = dif(y =  1)
                #    if bp[1] + 1 < R and ldone[bp[0], bp[1]  + 1, bp[2]] == 0:
                #        commands.append(SMove(d2))
                #        moved = True
                
            if not moved:
                player_broke[p] += 1
                if player_broke[p] > 5 : player_broke[p] = 0 
                i = int(random.random() * 3)
                j = int(random.random() * 2)
                s = 1 if j == 0 else -1
                d2 = dif(s if i == 0 else 0,
                         s if i == 1 else 0,
                         s if i == 2 else 0)
                #s = 1 if d2[i] > 0 else -1
                for x in range(0, abs(d2[i]) + 1):
                    d3 = [s* x + bp[0] if i == 0 else bp[0], 
                            s*x + bp[1]  if i == 1 else bp[1], 
                            s*x +  bp[2] if i == 2 else bp[2]]
                    if d3[0] < 0 or d3[0] >= R or d3[1] < 0 or d3[1] >= R or d3[2] < 0 or d3[2] >= R or ldone(d3[0], d3[1], d3[2]):
                        break
                    d2[i] = s * x 
                #assert d2[0] != 0
                if d2[i] != 0:
                    commands.append(SMove(d2))
                    moved = True
            #    d2 = dif(y = 1)
            #    commands.append(SMove(d2))
            #    moved = True
            if not moved:
                commands.append(Wait())
            #for x in range(bot_pos[0], bot_pos[0] + d[0]):
            #    assert done[x, d[1], d[2]] == 0, x
            #for z in range(bot_pos[2], bot_pos[2] + d[2]):
            #    assert done[d[0], d[1], z] == 0, z
            for j in range(mlen(d2) + 1):
                q = bp + (d2 // mlen(d2)) * j    
                
                inter[q[0], q[1], q[2]] =  1 
                #print(p, bot_pos[p], q, bot_pos[p] + d2)

            bot_pos[p] = bot_pos[p] + d2
            
            if counter % 30000 == 0: 
                print(p, bot_pos[p], t[p], player_done, commands[-1], rank, t[p][1])
                if t[p][1] == rank[p] and player_done[p] == False:
                    raise Exception
                rank[p] = t[p][1]
            inter[bot_pos[p][0], bot_pos[p][1], bot_pos[p][2]] = 1
            if counter > 5000000:
                raise Exception
        #assert len(commands) - start_com == len(bots_left)
        #print(bot_pos, t)
            #assert(moved)
            #print(bot_pos, t["pos"])
    return commands

def load_model(f="models/LA001_tgt.mdl"):
    i = 0
    c = [0, 0, 0]
    j = 0
    for i, b in enumerate(bytes_from_file(f)):
        if i == 0:
            R = b
            print(R)
            matrix = np.zeros([R+1, R+1, R+1], dtype=bool)
        else:
            bits = bin(b)[2:]
            #print(bits)
            bits = "0"* (8-len(bits)) + bits

            #print(bits)
            for b2 in reversed(bits):
                #print(b2)
                matrix[ j // (R*R), (j // R) % R , j % R ] = (b2 == "1") 
                j += 1
    return matrix, R



import random
for k in range(186, 187):
    print(k)
    print("load")
    model, R = load_model("models/LA%03d_tgt.mdl"%k)
    print("order")
    graph = order(model, R)
    print("order_targets")

    try:
        raise Exception
        targets = order_targets(graph, R, 7, [(0,0,0), (0, 1, 0),  (0, 0, 1), (1, 0, 0), (1,1,0), 
                                              (0, 1, 1), (1, 0, 1)])
        print("commands")
        commands = [Fission((0, 1, 0), 1)]
        commands += [Fission((0, 0, 1), 1)]
        commands += [Wait()]
        commands += [Fission((1, 0, 0), 1)]
        commands += [Wait()]
        commands += [Wait()]
        commands += [Fission((1, 1, 0), 1)]
        commands += [Wait()]
        commands += [Wait()]
        commands += [Wait()]
        commands += [Fission((0, 1, 1), 1)]
        commands += [Wait()]
        commands += [Wait()]
        commands += [Wait()]
        commands += [Wait()]
        commands += [Fission((1, 0, 1), 1)]
        commands += [Wait()]
        commands += [Wait()]
        commands += [Wait()]
        commands += [Wait()]
        commands += [Wait()]
        commandsa = make_commands(targets, R, [(0,0,0), (0, 1, 0),  (0, 0, 1), (1, 0, 0), (1,1,0), 
                                              (0, 1, 1), (1, 0, 1)])
        commands += commandsa
       
    except Exception:
        
        try:
            raise Exception
            targets = order_targets(graph, R, 4, [(0,0,0), (0, 1, 0),  (0, 0, 1), (1, 0, 0)])
            print("commands")
            commands = [Fission((0, 1, 0), 1)]
            commands += [Fission((0, 0, 1), 1)]
            commands += [Wait()]
            commands += [Fission((1, 0, 0), 1)]
            commands += [Wait()]
            commands += [Wait()]
            commandsa = make_commands(targets, R, [(0,0,0), (0, 1, 0), (0, 0, 1), (1 , 0, 0)])
            commands += commandsa

        except Exception: 
            try:
                raise Exception
                targets = order_targets(graph, R, 2, [(0,0,0), (0, 1, 0)])
                print("commands")
                commands = [Fission((0, 1, 0), 2)]
                commandsa = make_commands(targets, R, [(0,0,0), (0, 1, 0)])
                commands += commandsa
            except Exception:
                targets = order_targets(graph, R, 1, [(0,0,0)])
                print("commands")
                commands = []
                commandsa = make_commands(targets, R, [(0,0,0)])
                commands += commandsa
    #commands.append(Halt())
    write_commands(commands, "traces_my/LA%03d.nbt"%k)
    #print(targets)








def show(m):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_ylabel("y")
    ax.voxels(m)
    
class NanoState:
    @staticmethod
    def start(R):
        return NanoState(0, False, np.zeros([R, R, R]), [BotState(1, [0,0,0], range(2, 21))])
    
    def __init__(self, energy, harmonics, matrix, bots):
        self.energy = energy # int
        self.harmonics = harmonics # bool
        self.matrix = matrix # mat
        self.bots = bots # set
        
    def update(self):
        if self.harmonics:
            self.energy = self.energy + 30 * R * R * R
        else:
            self.energy = self.energy + 3 * R * R * R
        self.energy = self.energy + 20 * len(self.bots) 
    
    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = fig.gca(projection='3d')
        ax.set_ylabel("y")
        ax.voxels(self.matrix)
    
class BotState:
    def __init__(self, bid, pos, seeds):
        self.bid = bid # int
        self.pos = pos # coord
        self.seeds = seeds # set


# In[ ]:


