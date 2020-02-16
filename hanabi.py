import random
import sys
import copy
import time

GREEN = 0
YELLOW = 1
WHITE = 2
BLUE = 3
RED = 4
ALL_COLORS = [GREEN, YELLOW, WHITE, BLUE, RED]
COLORNAMES = ["green", "yellow", "white", "blue", "red"]

COUNTS = [3,2,2,2,1]  # num. of cards in game of each rank


def f(something):
    '''
    semi-intelligently format cards in any format. For printing, not important for game itself.
    :param something:
    :return:
    '''
    if type(something) == list:
        return map(f, something)
    elif type(something) == dict:
        return {k: something(v) for (k,v) in something.iteritems()}
    elif type(something) == tuple and len(something) == 2:
        return (COLORNAMES[something[0]],something[1])
    return something

def make_deck():
    '''
    Function to initalise the deck
    :return: list, deck
    '''
    deck = []
    for col in ALL_COLORS:
        for num, cnt in enumerate(COUNTS):
            for i in xrange(cnt):
                deck.append((col, num+1))
    random.seed()  # ADD: otherwise deck is always same?
    random.shuffle(deck)
    return deck
    
def initial_knowledge():
    '''
    initial common knowledge is just counts of all ranks (3,2,2,2,1)
    :return: list, knowledge
    '''
    knowledge = []
    for col in ALL_COLORS:
        knowledge.append(COUNTS[:])
    return knowledge
    
def hint_color(knowledge, color, truth):
    result = []
    for col in ALL_COLORS:
        if truth == (col == color):
            result.append(knowledge[col][:])
        else:
            result.append([0 for i in knowledge[col]])
    return result
    
def hint_rank(knowledge, rank, truth):
    result = []
    for col in ALL_COLORS:
        colknow = []
        for i,k in enumerate(knowledge[col]):
            if truth == (i + 1 == rank):
                colknow.append(k)
            else:
                colknow.append(0)
        result.append(colknow)
    return result
    
def iscard((c,n)):
    knowledge = []
    for col in ALL_COLORS:
        knowledge.append(COUNTS[:])
        for i in xrange(len(knowledge[-1])):
            if col != c or i+1 != n:
                knowledge[-1][i] = 0
            else:
                knowledge[-1][i] = 1
            
    return knowledge
    
# different actions are codified with integers
HINT_COLOR = 0
HINT_NUMBER = 1
PLAY = 2
DISCARD = 3
    
class Action(object):
    def __init__(self, type, pnr=None, col=None, num=None, cnr=None):
        self.type = type  # what kind of action?
        self.pnr = pnr  # player number
        self.col = col  # color
        self.num = num  # rank, 0-based
        self.cnr = cnr  # card number (n-th card), 0-based
    def __str__(self):
        if self.type == HINT_COLOR:
            return "hints " + str(self.pnr) + " about all their " + COLORNAMES[self.col] + " cards"
        if self.type == HINT_NUMBER:
            return "hints " + str(self.pnr) + " about all their " + str(self.num)
        if self.type == PLAY:
            return "plays their " + str(self.cnr)
        if self.type == DISCARD:
            return "discards their " + str(self.cnr)
    def __eq__(self, other):
        return (self.type, self.pnr, self.col, self.num, self.cnr) == (other.type, other.pnr, other.col, other.num, other.cnr)


class Player(object):
    def __init__(self, name, pnr):
        self.name = name
        self.explanation = []
    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        return random.choice(valid_actions)
    def inform(self, action, player, game):
        pass
    def get_explanation(self):
        return self.explanation


def get_possible(knowledge):
    '''
    Get all possibility of each card
    :param knowledge: nested list, knowledge of a single card of nr-th player
    :return: list of tuple, all possible combinations of color and rank
    '''
    result = []
    for col in ALL_COLORS:
        for rank, count in enumerate(knowledge[col]):  # for each [col][rank] combination
            if count > 0:  # if there are still cards left
                result.append((col,rank+1))
    return result


def playable(possible, board):
    '''
    return True if the card is surely playable
    :param possible:
    :param board:
    :return:
    '''
    for (col,nr) in possible:
        if board[col][1] + 1 != nr:
            return False
    return True


def potentially_playable(possible, board):
    '''
    return True if the card might be playable
    :param possible:
    :param board:
    :return:
    '''
    for (col,nr) in possible:
        if board[col][1] + 1 == nr:
            return True
    return False


def discardable(possible, board):
    '''
    return True if the card is surely discardable
    :param possible:
    :param board:
    :return:
    '''
    for (col,nr) in possible:
        if board[col][1] < nr:
            return False
    return True
    

def potentially_discardable(possible, board):
    '''
    return True if the card might be discardable
    :param possible:
    :param board:
    :return:
    '''
    for (col,nr) in possible:
        if board[col][1] >= nr:
            return True
    return False

# # ADD: reason about possiblity of keeping a card. This is not needed anymore
# def potentially_keep(possible, board, trash):
#     for (color,rank) in possible:
#         if board[color][1] < rank:  # the card is still useful
#             for (trashedcolor,trashedrank) in trash:  # keep the card if it has been already trashed
#                 if (color,rank) == (trashedcolor,trashedrank):
#                     print('match!:', color, rank)
#                     return True
#     return False


def update_knowledge(knowledge, used):
    '''
    update the knowledge structure of all agents
    :param knowledge:
    :param used:
    :return:
    '''
    result = copy.deepcopy(knowledge)
    for r in result:
        for (c,nr) in used:
            r[c][nr-1] = max(r[c][nr-1] - used[c,nr], 0)
    return result



############# AGENTS ############################
################################################


class InnerStatePlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        handsize = len(knowledge[0])
        possible = []
        for k in knowledge[nr]:
            possible.append(get_possible(k))

        discards = []
        duplicates = []
        for i, p in enumerate(possible):
            if playable(p, board):
                return Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        if discards:
            return Action(DISCARD, cnr=random.choice(discards))

        playables = []
        for i, h in enumerate(hands):
            if i != nr:
                for j, (col, n) in enumerate(h):
                    if board[col][1] + 1 == n:
                        playables.append((i, j))

        if playables and hints > 0:
            i, j = playables[0]
            if random.random() < 0.5:
                return Action(HINT_COLOR, pnr=i, col=hands[i][j][0])
            return Action(HINT_NUMBER, pnr=i, num=hands[i][j][1])

        for i, k in enumerate(knowledge):
            if i == nr:
                continue
            cards = range(len(k))
            random.shuffle(cards)
            c = cards[0]
            (col, num) = hands[i][c]
            hinttype = [HINT_COLOR, HINT_NUMBER]
            if hinttype and hints > 0:
                if random.choice(hinttype) == HINT_COLOR:
                    return Action(HINT_COLOR, pnr=i, col=col)
                else:
                    return Action(HINT_NUMBER, pnr=i, num=num)

        prefer = []
        for v in valid_actions:
            if v.type in [HINT_COLOR, HINT_NUMBER]:
                prefer.append(v)
        prefer = []
        if prefer and hints > 0:
            return random.choice(prefer)
        return random.choice([Action(DISCARD, cnr=i) for i in xrange(len(knowledge[0]))])

    def inform(self, action, player, game):
        pass


class OuterStatePlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.hints = {}
        self.pnr = pnr
        self.explanation = []
    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        handsize = len(knowledge[0])
        possible = []
        for k in knowledge[nr]:
            possible.append(get_possible(k))
        
        discards = []
        duplicates = []
        for i,p in enumerate(possible):
            if playable(p,board):
                return Action(PLAY, cnr=i)
            if discardable(p,board):
                discards.append(i)

        if discards:
            return Action(DISCARD, cnr=random.choice(discards))
            
        playables = []
        for i,h in enumerate(hands):
            if i != nr:
                for j,(col,n) in enumerate(h):
                    if board[col][1] + 1 == n:
                        playables.append((i,j))
        playables.sort(key=lambda (i,j): -hands[i][j][1])
        while playables and hints > 0:
            i,j = playables[0]
            knows_rank = True
            real_color = hands[i][j][0]
            real_rank = hands[i][j][0]
            k = knowledge[i][j]
            
            hinttype = [HINT_COLOR, HINT_NUMBER]
            if (j,i) not in self.hints:
                self.hints[(j,i)] = []
            
            for h in self.hints[(j,i)]:
                hinttype.remove(h)
            
            t = None
            if hinttype:
                t = random.choice(hinttype)
            
            if t == HINT_NUMBER:
                self.hints[(j,i)].append(HINT_NUMBER)
                return Action(HINT_NUMBER, pnr=i, num=hands[i][j][1])
            if t == HINT_COLOR:
                self.hints[(j,i)].append(HINT_COLOR)
                return Action(HINT_COLOR, pnr=i, col=hands[i][j][0])
            
            playables = playables[1:]
        
        for i, k in enumerate(knowledge):
            if i == nr:
                continue
            cards = range(len(k))
            random.shuffle(cards)
            c = cards[0]
            (col,num) = hands[i][c]            
            hinttype = [HINT_COLOR, HINT_NUMBER]
            if (c,i) not in self.hints:
                self.hints[(c,i)] = []
            for h in self.hints[(c,i)]:
                hinttype.remove(h)
            if hinttype and hints > 0:
                if random.choice(hinttype) == HINT_COLOR:
                    self.hints[(c,i)].append(HINT_COLOR)
                    return Action(HINT_COLOR, pnr=i, col=col)
                else:
                    self.hints[(c,i)].append(HINT_NUMBER)
                    return Action(HINT_NUMBER, pnr=i, num=num)

        return random.choice([Action(DISCARD, cnr=i) for i in xrange(handsize)])
    def inform(self, action, player, game):
        if action.type in [PLAY, DISCARD]:
            x = str(action)
            if (action.cnr,player) in self.hints:
                self.hints[(action.cnr,player)] = []
            for i in xrange(10):
                if (action.cnr+i+1,player) in self.hints:
                    self.hints[(action.cnr+i,player)] = self.hints[(action.cnr+i+1,player)]
                    self.hints[(action.cnr+i+1,player)] = []


def generate_hands(knowledge, used={}):
    if len(knowledge) == 0:
        yield []
        return

    for other in generate_hands(knowledge[1:], used):
        for col in ALL_COLORS:
            for i, cnt in enumerate(knowledge[0][col]):
                if cnt > 0:

                    result = [(col, i + 1)] + other
                    ok = True
                    thishand = {}
                    for (c, n) in result:
                        if (c, n) not in thishand:
                            thishand[(c, n)] = 0
                        thishand[(c, n)] += 1
                    for (c, n) in thishand:
                        if used[(c, n)] + thishand[(c, n)] > COUNTS[n - 1]:
                            ok = False
                    if ok:
                        yield result


def generate_hands_simple(knowledge, used={}):
    if len(knowledge) == 0:
        yield []
        return
    for other in generate_hands_simple(knowledge[1:]):
        for col in ALL_COLORS:
            for i, cnt in enumerate(knowledge[0][col]):
                if cnt > 0:
                    yield [(col, i + 1)] + other


a = 1


class SelfRecognitionPlayer(Player):
    def __init__(self, name, pnr, other=OuterStatePlayer):
        self.name = name
        self.hints = {}
        self.pnr = pnr
        self.gothint = None
        self.last_knowledge = []
        self.last_played = []
        self.last_board = []
        self.other = other
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        handsize = len(knowledge[0])
        possible = []

        if self.gothint:

            possiblehands = []
            wrong = 0
            used = {}
            for c in ALL_COLORS:
                for i, cnt in enumerate(COUNTS):
                    used[(c, i + 1)] = 0
            for c in trash + played:
                used[c] += 1

            for h in generate_hands_simple(knowledge[nr], used):
                newhands = hands[:]
                newhands[nr] = h
                other = self.other("Pinocchio", self.gothint[1])
                act = other.get_action(self.gothint[1], newhands, self.last_knowledge, self.last_trash,
                                       self.last_played, self.last_board, valid_actions, hints + 1)
                lastact = self.gothint[0]
                if act == lastact:
                    possiblehands.append(h)

                    def do(c, i):
                        newhands = hands[:]
                        h1 = h[:]
                        h1[i] = c
                        newhands[nr] = h1
                        print other.get_action(self.gothint[1], newhands, self.last_knowledge, self.last_trash,
                                               self.last_played, self.last_board, valid_actions, hints + 1)
                    # import pdb
                    # pdb.set_trace()
                else:
                    wrong += 1
            # print len(possiblehands), "would have led to", self.gothint[0], "and not:", wrong
            # print f(possiblehands)
            if possiblehands:
                mostlikely = [(0, 0) for i in xrange(len(possiblehands[0]))]
                for i in xrange(len(possiblehands[0])):
                    counts = {}
                    for h in possiblehands:
                        if h[i] not in counts:
                            counts[h[i]] = 0
                        counts[h[i]] += 1
                    for c in counts:
                        if counts[c] > mostlikely[i][1]:
                            mostlikely[i] = (c, counts[c])
                # print "most likely:", mostlikely
                m = max(mostlikely, key=lambda (card, cnt): cnt)
                second = mostlikely[:]
                second.remove(m)
                m2 = max(second, key=lambda (card, cnt): cnt)
                if m[1] >= m2[1] * a:
                    # print ">>>>>>> deduced!", f(m[0]), m[1],"vs", f(m2[0]), m2[1]
                    knowledge = copy.deepcopy(knowledge)
                    knowledge[nr][mostlikely.index(m)] = iscard(m[0])

        self.gothint = None
        for k in knowledge[nr]:
            possible.append(get_possible(k))

        discards = []
        duplicates = []
        for i, p in enumerate(possible):
            if playable(p, board):
                return Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        if discards:
            return Action(DISCARD, cnr=random.choice(discards))

        playables = []
        for i, h in enumerate(hands):
            if i != nr:
                for j, (col, n) in enumerate(h):
                    if board[col][1] + 1 == n:
                        playables.append((i, j))
        playables.sort(key=lambda (i, j): -hands[i][j][1])
        while playables and hints > 0:
            i, j = playables[0]
            knows_rank = True
            real_color = hands[i][j][0]
            real_rank = hands[i][j][0]
            k = knowledge[i][j]

            hinttype = [HINT_COLOR, HINT_NUMBER]
            if (j, i) not in self.hints:
                self.hints[(j, i)] = []

            for h in self.hints[(j, i)]:
                hinttype.remove(h)

            if HINT_NUMBER in hinttype:
                self.hints[(j, i)].append(HINT_NUMBER)
                return Action(HINT_NUMBER, pnr=i, num=hands[i][j][1])
            if HINT_COLOR in hinttype:
                self.hints[(j, i)].append(HINT_COLOR)
                return Action(HINT_COLOR, pnr=i, col=hands[i][j][0])

            playables = playables[1:]

        for i, k in enumerate(knowledge):
            if i == nr:
                continue
            cards = range(len(k))
            random.shuffle(cards)
            c = cards[0]
            (col, num) = hands[i][c]
            hinttype = [HINT_COLOR, HINT_NUMBER]
            if (c, i) not in self.hints:
                self.hints[(c, i)] = []
            for h in self.hints[(c, i)]:
                hinttype.remove(h)
            if hinttype and hints > 0:
                if random.choice(hinttype) == HINT_COLOR:
                    self.hints[(c, i)].append(HINT_COLOR)
                    return Action(HINT_COLOR, pnr=i, col=col)
                else:
                    self.hints[(c, i)].append(HINT_NUMBER)
                    return Action(HINT_NUMBER, pnr=i, num=num)

        return random.choice([Action(DISCARD, cnr=i) for i in xrange(handsize)])

    def inform(self, action, player, game):
        if action.type in [PLAY, DISCARD]:
            x = str(action)
            if (action.cnr, player) in self.hints:
                self.hints[(action.cnr, player)] = []
            for i in xrange(10):
                if (action.cnr + i + 1, player) in self.hints:
                    self.hints[(action.cnr + i, player)] = self.hints[(action.cnr + i + 1, player)]
                    self.hints[(action.cnr + i + 1, player)] = []
        elif action.pnr == self.pnr:
            self.gothint = (action, player)
            self.last_knowledge = game.knowledge[:]
            self.last_board = game.board[:]
            self.last_trash = game.trash[:]
            self.played = game.played[:]


TIMESCALE = 40.0 / 1000.0  # ms
SLICETIME = TIMESCALE / 10.0
APPROXTIME = SLICETIME / 8.0


def priorities(c, board):
    (col, val) = c
    if board[col][1] == val - 1:
        return val - 1
    if board[col][1] >= val:
        return 5
    if val == 5:
        return 15
    return 6 + (4 - val)


SENT = 0
ERRORS = 0
COUNT = 0

CAREFUL = True


class TimedPlayer(object):
    def __init__(self, name, pnr):
        self.name = name
        self.explanation = []
        self.last_tick = time.time()
        self.pnr = pnr
        self.last_played = False
        self.tt = time.time()

    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        global SENT, ERRORS, COUNT
        tick = time.time()
        duration = round((tick - self.last_tick) / SLICETIME)
        other = (self.pnr + 1) % len(hands)
        # print(self.pnr, "got", duration)
        if duration >= 10:
            duration = 9
        if duration != SENT:
            ERRORS += 1
            # print("mismatch", nr, f(hands), f(board), duration, SENT)
        COUNT += 1
        other_hand = hands[other][:]

        def prio(c):
            return priorities(c, board)

        other_hand.sort(key=prio)
        # print(f(other_hand), f(board), list(map(prio, other_hand)), f(hands))
        p = prio(other_hand[0])
        delta = 0.0
        if p >= 5:
            delta += 5

        # print("idx", hands[other].index(other_hand[0]))
        def fix(n):
            if n >= len(other_hand):
                return len(other_hand) - 1
            return int(round(n))

        delta += hands[other].index(other_hand[0])
        if duration >= 5:
            action = Action(DISCARD, cnr=fix(duration - 5))
        else:
            action = Action(PLAY, cnr=fix(duration))
        if self.last_played and hints > 0 and CAREFUL:
            action = Action(HINT_COLOR, pnr=other, col=other_hand[0][0])
        t1 = time.time()
        SENT = delta
        # print(self.pnr, "convey", round(delta))
        delta -= 0.5
        while (t1 - tick) < delta * SLICETIME:
            time.sleep(APPROXTIME)
            t1 = time.time()
        self.last_tick = time.time()
        return action

    def inform(self, action, player, game):
        self.last_played = (action.type == PLAY)
        self.last_tick = self.tt
        self.tt = time.time()
        # print(action, player)

    def get_explanation(self):
        return self.explanation


CANDISCARD = 128


def format_intention(i):
    '''
    for pretty printin
    :param i:
    :return:
    '''
    if isinstance(i, str):
        return i
    if i == PLAY:
        return "Play"
    elif i == DISCARD:
        return "Discard"
    elif i == CANDISCARD:
        return "Can Discard"
    return "Keep"
    

def whattodo(knowledge, pointed, board):
    '''
    This is not needed anymore because I explicitly implemented this.
    The idea is to reason whattodo given a hint
    :param knowledge:
    :param pointed:
    :param board:
    :return:
    '''
# @pointed: whether my card is postively identified and there is a possibility of my card being that
# e.g.) given a hint about all cards that are red, whether my second card can be red
    possible = get_possible(knowledge)
    play = potentially_playable(possible, board)
    discard = potentially_discardable(possible, board)

    if play and pointed:  # if I can play the card and I possibly have that card
        return PLAY
    if discard and pointed:
        return DISCARD
    return None

def pretend(action, knowledge, intentions, hand, board):
    '''
    predict the action of the other player given my action and how good it is
    :param action: tuple, (type,value)
    :param knowledge: nested list, knowledge of [1-nr] player (so the other player)
    :param intentions: list, my inferred intention for all players
    :param hand: list, hand of 1-nr-th player
    :param board: top cards
    :return: (bool=isvalid, int=score, expl=list prediction of other players action)
    '''

    (type,value) = action  # type; color or rank, value; the actual value (e.g. red ...)
    positive = []
    haspositive = False  # True if some card is positively identified (e.g.told that it is red) by the hint
    change = False
    if type == HINT_COLOR:
        newknowledge = []  # M'_B in paper
        for i,(col,num) in enumerate(hand):  # color and rank of each i-th card
            positive.append(value==col)  # which cards are actually red?
            newknowledge.append(hint_color(knowledge[i], value, value == col))
            if value == col:
                haspositive = True
                if newknowledge[-1] != knowledge[i]:
                    change = True
    else:  # rank hint, analog
        newknowledge = []
        for i,(col,num) in enumerate(hand):
            positive.append(value==num)
            
            newknowledge.append(hint_rank(knowledge[i], value, value == num))
            if value == num:
                haspositive = True
                if newknowledge[-1] != knowledge[i]:
                    change = True
    if not haspositive:
        return False, 0, ["Invalid hint"]
    if not change:
        return False, 0, ["No new information"]
    score = 0
    predictions = []
    pos = False
    for i,c,k,p in zip(intentions, hand, newknowledge, positive):
        
        action = whattodo(k, p, board)  # assume that the co-player will follow my logic to choose her action!
        
        if action == PLAY and i != PLAY:
            #print "would cause them to play", f(c)
            return False, 0, predictions + [PLAY]
        
        if action == DISCARD and i not in [DISCARD, CANDISCARD]:
            #print "would cause them to discard", f(c)
            return False, 0, predictions + [DISCARD]
            
        if action == PLAY and i == PLAY:
            pos = True
            predictions.append(PLAY)
            score += 3
        elif action == DISCARD and i in [DISCARD, CANDISCARD]:
            pos = True
            predictions.append(DISCARD)
            if i == DISCARD:
                score += 2
            else:
                score += 1
        else:
            predictions.append(None)
    if not pos:
        return False, score, predictions
    return True,score, predictions
    
HINT_VALUE = 0.5  # hint is worth half a point in maydiscard decision, but this is not needed anymore


def pretend_discard(act, knowledge, board, trash):
    '''
    It tries to compute the expected loss of discarding each card.
    This will be not necessary anymore, I propose to simply discard the oldest card.
    :param act:
    :param knowledge:
    :param board:
    :param trash:
    :return:
    '''
    which = copy.deepcopy(knowledge[act.cnr])
    for (col,num) in trash:
        if which[col][num-1]:  # if the same type of card is already trashed, better not throw another one awy
            which[col][num-1] -= 1
    for col in ALL_COLORS:
        for i in xrange(board[col][1]):  # for the rank of each color
            if which[col][i]:  # if the card can be played, rather not throw it
                which[col][i] -= 1
    possibilities = sum(map(sum, which))  # normalization constant
    expected = 0  # expected value of each possible discarding
    terms = []
    for col in ALL_COLORS:
        for i,cnt in enumerate(which[col]):
            rank = i+1  # index shift
            if cnt > 0:  # if I still have card left of 'col' and 'rank'
                prob = cnt*1.0/possibilities  # more likely to discard if more cards left
                if board[col][1] >= rank:  # if this specific potential realization is not needed anymore
                    expected += prob*HINT_VALUE
                    terms.append((col,rank,cnt,prob,prob*HINT_VALUE))
                else:  # if card is still needed
                    dist = rank - board[col][1]  # how relevant is it in the near future?
                    if cnt > 1:
                        value = prob*(6-rank)/(dist*dist)
                    else:
                        value = (6-rank)
                    if rank == 5:  # take into account that you win a value of getting back a hint token by playing 5
                        value += HINT_VALUE
                    value *= prob
                    expected -= value
                    terms.append((col,rank,cnt,prob,-value))
    return (act, expected, terms)


def format_knowledge(k):
    '''
    for pretty printing
    :param k:
    :return:
    '''
    result = ""
    for col in ALL_COLORS:
        for i,cnt in enumerate(k[col]):
            if cnt > 0:
                result += COLORNAMES[col] + " " + str(i+1) + ": " + str(cnt) + "\n"
    return result


class IntentionalPlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.hints = {}
        self.pnr = pnr
        self.gothint = None
        self.last_knowledge = []
        self.last_played = []
        self.last_board = []
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        handsize = len(knowledge[0])
        possible = []
        result = None
        self.explanation = []
        self.explanation.append(["Your Hand:"] + map(f, hands[1 - nr]))

        self.gothint = None
        for k in knowledge[nr]:
            possible.append(get_possible(k))

        discards = []
        duplicates = []
        for i, p in enumerate(possible):
            if playable(p, board):
                result = Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        if discards and hints < 8 and not result:
            result = Action(DISCARD, cnr=random.choice(discards))

        playables = []
        useless = []
        discardables = []
        othercards = trash + board
        intentions = [None for i in xrange(handsize)]
        for i, h in enumerate(hands):
            if i != nr:
                for j, (col, n) in enumerate(h):
                    if board[col][1] + 1 == n:
                        playables.append((i, j))
                        intentions[j] = PLAY
                    if board[col][1] >= n:
                        useless.append((i, j))
                        if not intentions[j]:
                            intentions[j] = DISCARD
                    if n < 5 and (col, n) not in othercards:
                        discardables.append((i, j))
                        if not intentions[j]:
                            intentions[j] = CANDISCARD

        self.explanation.append(["Intentions"] + map(format_intention, intentions))

        if hints > 0:
            valid = []
            for c in ALL_COLORS:
                action = (HINT_COLOR, c)
                # print "HINT", COLORNAMES[c],
                (isvalid, score, expl) = pretend(action, knowledge[1 - nr], intentions, hands[1 - nr], board)
                self.explanation.append(["Prediction for: Hint Color " + COLORNAMES[c]] + map(format_intention, expl))
                # print isvalid, score
                if isvalid:
                    valid.append((action, score))

            for r in xrange(5):
                r += 1
                action = (HINT_NUMBER, r)
                # print "HINT", r,

                (isvalid, score, expl) = pretend(action, knowledge[1 - nr], intentions, hands[1 - nr], board)
                self.explanation.append(["Prediction for: Hint Rank " + str(r)] + map(format_intention, expl))
                # print isvalid, score
                if isvalid:
                    valid.append((action, score))

            if valid and not result:
                valid.sort(key=lambda (a, s): -s)
                # print valid
                (a, s) = valid[0]
                if a[0] == HINT_COLOR:
                    result = Action(HINT_COLOR, pnr=1 - nr, col=a[1])
                else:
                    result = Action(HINT_NUMBER, pnr=1 - nr, num=a[1])

        self.explanation.append(["My Knowledge"] + map(format_knowledge, knowledge[nr]))
        possible = [Action(DISCARD, cnr=i) for i in xrange(handsize)]

        scores = map(lambda p: pretend_discard(p, knowledge[nr], board, trash), possible)

        def format_term((col, rank, n, prob, val)):
            return COLORNAMES[col] + " " + str(rank) + " (%.2f%%): %.2f" % (prob * 100, val)

        self.explanation.append(
            ["Discard Scores"] + map(lambda (a, s, t): "\n".join(map(format_term, t)) + "\n%.2f" % (s), scores))
        scores.sort(key=lambda (a, s, t): -s)
        if result:
            return result
        return scores[0][0]

        return random.choice([Action(DISCARD, cnr=i) for i in xrange(handsize)])

    def inform(self, action, player, game):
        if action.type in [PLAY, DISCARD]:
            x = str(action)
            if (action.cnr, player) in self.hints:
                self.hints[(action.cnr, player)] = []
            for i in xrange(10):
                if (action.cnr + i + 1, player) in self.hints:
                    self.hints[(action.cnr + i, player)] = self.hints[(action.cnr + i + 1, player)]
                    self.hints[(action.cnr + i + 1, player)] = []
        elif action.pnr == self.pnr:
            self.gothint = (action, player)
            self.last_knowledge = game.knowledge[:]
            self.last_board = game.board[:]
            self.last_trash = game.trash[:]
            self.played = game.played[:]


# ADD
def shift_index(acted_index, keeplist):
    '''
    aux function to help with shifting keeplist if a card is played, discarded ...
    :param acted_index: int, 0-based index of the card that is acted upon
    : param keeplist: set, set of index of cards to be kept
    :return: None
    '''
    new_set = copy.deepcopy(keeplist)
    for cnr in keeplist:
        if cnr >= acted_index:
            new_set.remove(cnr)
            if cnr > acted_index:
                new_set.add(cnr - 1)  # shift all cards that had higher or same index to the played index
    keeplist = copy.deepcopy(new_set)
    return keeplist


###### !!!!!!!!! THIS IS THE SHIT !!!!!!!!! #
##############################################
class SelfIntentionalPlayer(Player):

    def __init__(self, name, pnr):
        self.name = name
        self.hints = {}
        self.pnr = pnr  # player number
        self.gothint = None  # boolean, True iff hinted last round
        self.last_knowledge = []
        self.last_played = []
        self.last_board = []
        self.explanation = []
        self.keeplist = set()  # ADD: add a set of cards that shouldn't be discarded


    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        '''
        choose the action of this agent.
        :param nr: int, nr-th player
        :param hands: list of list, hands[nr]
        :param knowledge: nested list, knowledge[nr][i-th card][Color][rank] contains probability
        :param trash: list, discarded cards
        :param played: list, all cards played successfully
        :param board: list, convenience param showing top cards of played
        :param valid_actions: list, all possible actions, no hints action given no hint tokens
        :param hints: int, number of hint tokens left
        :return: tuple, result(of class Action) and score (int)
        '''

        ##### reason about possible hands of yours, return possible #####
        handsize = len(knowledge[0])  # all player have same handsize so just take the handsize of first player
        possible = []  # list of list of tuples, possible[card][possibilities] = (color,rank)
        result = None  # What will I do? PLAY, DISCARD, HINT...
        self.explanation = []  # Text to be shown in UI
        self.explanation.append(["Your Hand:"] + map(f, hands[1-nr]))

        ##### (1) do what the other player (human) wants me to do #####
        # ### 1A) original ###
        # action = []  # list, list of all plausible action
        # if self.gothint:
        #     (act, plr) = self.gothint
        #     if act.type == HINT_COLOR:
        #         for k in knowledge[nr]:
        #             action.append(whattodo(k,
        #                                    sum(k[act.col]) > 0,  # if positively identified
        #                                    # e.g.) I am given a hint about red cards, and my ci-th card is possibly red
        #                                    board))
        #     elif act.type == HINT_NUMBER:
        #         for k in knowledge[nr]:
        #             cnt = 0
        #             for c in ALL_COLORS:
        #                 cnt += k[c][act.num - 1]
        #             action.append(whattodo(k, cnt > 0, board))
        #
        # if action:
        #     self.explanation.append(["What you want me to do"] + map(format_intention, action))
        #     for i, a in enumerate(action):  # play if it's possible to play, otherwise discard
        #         if a == PLAY and (not result or result.type == DISCARD):
        #             result = Action(PLAY, cnr=i)
        #         elif a == DISCARD and not result:
        #             result = Action(DISCARD, cnr=i)
        #######################################################################################


        # ### 1B) ADD: play conservatively ###
        # # Try to keep all cards that are hinted, and play only when it's sure
        # action = None  # ADD: just keep one possible action and not a list of whole possible actions
        # card_index = None
        # if self.gothint:  # if I am given a hint about my hands
        #     (act,plr) = self.gothint
        #     if act.type == HINT_COLOR:
        #         for ci, card in enumerate(knowledge[nr]):
        #             pointed = sum(card[act.col]) > 0
        #             possible_hint = get_possible(card)
        #             play = playable(possible_hint, board)  # should play the surely playable card if possible
        #             discard = discardable(possible_hint, board)  # and only surely discardable
        #             if play and pointed:
        #                 action = PLAY  # the last (the newest) surely playable card should be played
        #                 card_index = ci
        #             elif discard and pointed and action != PLAY:  # discard only if no card is playable
        #                 action = DISCARD
        #                 card_index = ci
        #     elif act.type == HINT_NUMBER:  # analog to color hint
        #         for ci, card in enumerate(knowledge[nr]):
        #             cnt = 0
        #             for c in ALL_COLORS:
        #                 cnt += card[c][act.num-1]
        #             pointed = cnt > 0
        #             possible_hint = get_possible(card)
        #             play = playable(possible_hint, board)
        #             discard = discardable(possible_hint, board)
        #             if play and pointed:
        #                 action = PLAY
        #                 card_index = ci
        #             elif discard and pointed and action != PLAY:
        #                 action = DISCARD
        #                 card_index = ci
        #
        #     # when the hint doesn't lead to any surely possible actions
        #     if action is None:
        #         print('KEEP function activated')
        #         if act.type == HINT_COLOR:
        #             for ci, card in enumerate(knowledge[nr]):
        #                 pointed = sum(card[act.col]) > 0
        #                 if pointed:
        #                     self.keeplist.add(ci)
        #         elif act.type == HINT_NUMBER:
        #             for ci,card in enumerate(knowledge[nr]):
        #                 cnt = 0
        #                 for c in ALL_COLORS:
        #                     cnt += card[c][act.num - 1]
        #                 pointed = cnt > 0
        #                 if pointed:
        #                     self.keeplist.add(ci)
        #     else:
        #         result = Action(action, cnr=card_index)
        #         shift_index(acted_index=card_index)
        #
        # #  this is now deprecated. The idea was to keep the card only if
        # #  there is exactly one card of that specific type left
        # #     # ADD: KEEP if no hints about play or discard is given
        # #     # if (len(set(action)) == 1) and (action[0] is None):
        # #     if action is None:
        # #         print('KEEP option activated')
        # #         if act.type == HINT_COLOR:
        # #             for ci, card in enumerate(knowledge[nr]):
        # #                 pointed = sum(card[act.col]) > 0
        # #                 if pointed:
        # #                     print('pointed')
        # #                     for col in ALL_COLORS:
        # #                         for rank in range(5):
        # #                             if card[col][rank] == 1:  # exactly one card left of that specific sort
        # #                                 self.keeplist.add(ci)
        # #                                 print('keep {}th card given color hint, card might be col:{} rank:{}'.format(
        # #                                     ci+1, COLORNAMES[col], rank+1), self.keeplist)
        # #                             else:
        # #                                 print(ci+1, COLORNAMES[col], rank+1, card[col][rank])
        # #         elif act.type == HINT_NUMBER:
        # #             for ci, card in enumerate(knowledge[nr]):
        # #                 cnt = 0
        # #                 for c in ALL_COLORS:
        # #                     cnt += card[c][act.num - 1]
        # #                 pointed = cnt > 0
        # #                 if pointed:
        # #                     print('pointed')
        # #                     for col in ALL_COLORS:
        # #                         for rank in range(5):
        # #                             if card[col][rank] == 1:
        # #                                 self.keeplist.add(ci)
        # #                                 print('keep {}th card given num hint, card might be col:{} rank:{}'.format(
        # #                                     ci+1, COLORNAMES[col], rank+1), self.keeplist)
        # #                             else:
        # #                                 print(ci+1, COLORNAMES[col], rank+1, card[col][rank])
        # self.explanation.append(["What you want me to do"] + map(format_intention, action))
        # # this is now deprecated, needed when I have a list of actions
        # # for i,a in enumerate(action):
        # #     if a == PLAY and (not result or result.type == DISCARD):  # playing is preferred over discarding
        # #         result = Action(PLAY, cnr=i)
        # #         shift_index(i)
        # #         print('shift keeplist after hintedplay: ', self.keeplist)
        # #     elif a == DISCARD and not result:  # rather discard the first card
        # #         result = Action(DISCARD, cnr=i)
        # #         shift_index(i)
        # #         print('shift keeplist after hinteddiscard: ', self.keeplist)
        # ###############################################################################


        # ### 1C) ADD: play a bit more aggressively ###
        # # prioritise playing if there is any possibility to play a card
        # action = None
        # card_index = None
        #
        # if self.gothint:  # if I am given a hint about my hands
        #     (act, plr) = self.gothint
        #     if act.type == HINT_COLOR:
        #         for ci, card in enumerate(knowledge[nr]):
        #             pointed = sum(card[act.col]) > 0
        #             possible_hint = get_possible(card)
        #             play = playable(possible_hint, board)
        #             mayplay = potentially_playable(possible_hint, board)  # possibility of playing is prioritised
        #             discard = discardable(possible_hint, board)
        #             if play and pointed:
        #                 action = PLAY  # the last (the newest) surely playable card should be played
        #                 card_index = ci
        #             elif mayplay and pointed and action != PLAY:
        #                 action = PLAY  # keep action if it's play for sure
        #                 card_index = ci
        #             elif discard and pointed and action != PLAY:
        #                 action = DISCARD
        #                 card_index = ci
        #     elif act.type == HINT_NUMBER:  # analog to color hint
        #         for ci, card in enumerate(knowledge[nr]):
        #             cnt = 0
        #             for c in ALL_COLORS:
        #                 cnt += card[c][act.num - 1]
        #             pointed = cnt > 0
        #             possible_hint = get_possible(card)
        #             play = playable(possible_hint, board)
        #             mayplay = potentially_playable(possible_hint, board)
        #             discard = discardable(possible_hint, board)
        #             if play and pointed:
        #                 action = PLAY
        #                 card_index = ci
        #             elif mayplay and pointed and action != PLAY:
        #                 action = PLAY  # keep action if it's play for sure
        #                 card_index = ci
        #             elif discard and pointed and action != PLAY:
        #                 action = DISCARD
        #                 card_index = ci
        #
        #     # when the hint doesn't lead to any possible actions
        #     if action is None:
        #         print('KEEP function activated')
        #         if act.type == HINT_COLOR:
        #             for ci, card in enumerate(knowledge[nr]):
        #                 pointed = sum(card[act.col]) > 0
        #                 if pointed:
        #                     self.keeplist.add(ci)
        #         elif act.type == HINT_NUMBER:
        #             for ci, card in enumerate(knowledge[nr]):
        #                 cnt = 0
        #                 for c in ALL_COLORS:
        #                     cnt += card[c][act.num - 1]
        #                 pointed = cnt > 0
        #                 if pointed:
        #                     self.keeplist.add(ci)
        #     else:
        #         result = Action(action, cnr=card_index)
        #         self.keeplist = shift_index(acted_index=card_index, keeplist=self.keeplist)
        # ##############################################################################


        ### 1D) TODO: heuristically favour interpreting hints as keep sign for older cards
        ###############################################################################
        action = None  # PLAY, DISCARD
        card_index = None  # index of the card to be played or discarded

        if self.gothint:  # if I am given a hint about my hands
            (act, plr) = self.gothint
            if act.type == HINT_COLOR:
                for ci, card in enumerate(knowledge[nr]):
                    pointed = sum(card[act.col]) > 0
                    possible_hint = get_possible(card)
                    play = playable(possible_hint, board)
                    mayplay = potentially_playable(possible_hint, board)  # possibility of playing is prioritised
                    discard = discardable(possible_hint, board)
                    if play and pointed:
                        action = PLAY  # the last (the newest) surely playable card should be played
                        card_index = ci
                    elif mayplay and pointed and action != PLAY:
                        action = PLAY  # keep action if it's play for sure
                        card_index = ci
                    elif discard and pointed and action != PLAY:
                        action = DISCARD
                        card_index = ci


            elif act.type == HINT_NUMBER:  # analog to color hint
                # sure_played = False  # ADD: to play the oldest surely playable card if there are multiples
                play_set = set()
                mayplay_set = set()
                keep_set = set()
                discard_set = set()
                for ci, card in enumerate(knowledge[nr]):
                    cnt = 0
                    for c in ALL_COLORS:
                        cnt += card[c][act.num - 1]
                    pointed = cnt > 0

                    possible_hint = get_possible(card)
                    play = playable(possible_hint, board)
                    # ADD: Keep
                    keep = False
                    if pointed:
                        for col in ALL_COLORS:
                            for rank in range(5):
                                if card[col][rank] == 1:
                                    # print('keep {}th card given num hint, card might be col:{} rank:{}'.format(
                                    #     ci+1, COLORNAMES[col], rank+1), self.keeplist)
                                    keep = True
                                # else:
                                #     print(ci+1, COLORNAMES[col], rank+1, card[col][rank])
                    mayplay = potentially_playable(possible_hint, board)
                    discard = discardable(possible_hint, board)

                    # TODO: heuristics between mayplay and keep
                    # If multiple cards are positively identified: The card that is surely playable should be played,
                    # and other cards should be kept
                    if play and pointed:
                        play_set.add(ci)
                    elif mayplay and pointed:
                        mayplay_set.add(ci)
                    if keep and pointed:
                        keep_set.add(ci)
                    if discard and pointed:
                        discard_set.add(ci)

                    print('play, mayplay, keep, discard: ', play_set, mayplay_set, keep_set, discard_set)
                    if len(play_set):  # if there exists a playable card
                        action = PLAY
                        card_index = min(play_set)  # play the oldest playable and keep all others
                        # TODO: unhashable type set
                        self.keeplist = self.keeplist.union(play_set.union(mayplay_set).union(keep_set).difference(set([card_index])))
                        print('play a playable, keep others')
                    elif len(mayplay_set):  # if no cards are playable, play mayplayable
                        if min(mayplay_set) >= 2:  # only relatively new cards are mayplayable
                            action = PLAY
                            card_index = min(mayplay_set)
                            self.keeplist = self.keeplist.union(mayplay_set.union(keep_set).difference(set([card_index])))
                            print('play a mayplayable that is new')
                        elif len(mayplay_set.difference(keep_set)):  # cards that are only mayplayable and not keep
                            action = PLAY
                            card_index = min(mayplay_set.difference(keep_set))
                            self.keeplist = self.keeplist.union(mayplay_set.union(keep_set).difference(set([card_index])))
                            print('play a mayplayable that is not hinted to be kept')
                        else: # mayplay_set == keep_set
                            self.keeplist = self.keeplist.union(keep_set)
                            print('all mayplayable are also to be kept', mayplay_set == keep_set)
                    elif len(discard_set):  # only if all other options are gone, discard safely
                        action = DISCARD
                        card_index = random.choice(discard_set)
                        self.keeplist = self.keeplist.union(keep_set.difference(discard_set))
                    elif len(keep_set):  # only keep_set is not empty
                        self.keeplist = self.keeplist.union(keep_set)
                    else:
                        print('you moron')


                    # if play and pointed:
                    #     if not sure_played:  # this is the oldest surely playable card
                    #         action = PLAY
                    #         card_index = ci
                    #         print('PLAY: the first playable')
                    #     else:  # although this card is also playable, the older one is first played
                    #         self.keeplist.add(ci)
                    #         print('KEEP: playable but another playable')
                    # elif mayplay and keep and pointed:  # when a card is both pot.play. and keep., heurstics
                    #     if ci < 2:  # if cards are old, it is more likely to be hinted to keep to save it
                    #         self.keeplist.add(ci)
                    #         print('KEEP: old cards are more likely to be hinted to keep')
                    #     else:
                    #         if action != PLAY:
                    #             action = PLAY
                    #             card_index = ci
                    #             print('PLAY: no playable, card is new enough')
                    #         else:  # Not sure what this might be, rather keep it
                    #             print('KEEP: unsure')
                    # elif mayplay and pointed:
                    #     if action != PLAY:
                    #         action = PLAY
                    #         card_index = ci
                    #         print('PLAY: mayplayable when no surely playable')
                    #     else:
                    #         self.keeplist.add(ci)
                    #         print('KEEP: mayplayable but other card already to be played')
                    # elif keep:
                    #     self.keeplist.add(ci)
                    #     print('KEEP: hinted to be kept')
                    # elif discard and pointed and action != PLAY:
                    #     action = DISCARD
                    #     card_index = ci
                    #     print('DISCARD')

            if action is not None:
                result = Action(action, cnr=card_index)
                self.keeplist = shift_index(acted_index=card_index, keeplist=self.keeplist)
        ###############################################################################


        ##### infer my hands #####
        self.gothint = None
        for card in knowledge[nr]:
            possible.append(get_possible(card)) # all possibilities of my hands
        ###########################

        ##### 2,3) decide to play or discard #####
        # ### A: original ###
        # discards = []  # list of all useless cards
        # for i,card in enumerate(possible):  # for each card in hands
        #     if playable(card, board) and not result:
        #         print('i-th card is surely playable: ', i)
        #         result = Action(PLAY, cnr=i)  # surely playable card
        #     if discardable(card, board):
        #         discards.append(i)  # surely discardable cards
        #
        # if discards and hints < 8 and not result:  # discard if no card is playable and hint token is not max
        #     ci = random.choice(discards)
        #     result = Action(DISCARD, cnr=ci)
        # ##########################################################


        ### B: minor changes due to shifting and so on ###
        discards = []  # list of all useless cards
        if not result:  # no action from direct hints in stage 1
            for i, card in enumerate(possible):  # for each card in hands
                if playable(card, board):
                    print('i-th card is surely playable: ', i)
                    result = Action(PLAY, cnr=i)  # should play the newest card if possible
                    self.keeplist = shift_index(i, self.keeplist)  # shift so that keeplist makes sense
                    print('shift keeplist after play: ', self.keeplist)
                elif discardable(card, board):
                    discards.append(i)  # surely discardable cards

        if discards and hints < 8 and not result:  # discard if no card is playable and hint token is not max
            ci = random.choice(discards)
            result = Action(DISCARD, cnr=ci)
            self.keeplist = shift_index(ci, self.keeplist)
            print('shift keeplist after discard: ', self.keeplist)
        ##########################################################
        ###############################################################################


        ##### 4-1) Hint, CalculateGoals: what should other players (human) do ? #####
        playables = []  # playables = [(0,2)] 2nd card of first player can be played
        useless = []
        discardables = []
        othercards = trash + board
        intentions = [None for i in xrange(handsize)]  # for printing
        for i,h in enumerate(hands):  # hand of i-th player
            if i != nr:  # no need to infer intention of my own (I just know it)
                for j, (color, rank) in enumerate(h):  # j+1-th card of i+1-th players hand
                    if board[color][1] + 1 == rank:  # if rank of the card is exactly one more than board, it's playable
                        playables.append((i,j))  # j-th card of i-th player can be played
                        intentions[j] = PLAY  # then j+1-th card should be played
                    if board[color][1] >= rank:  # rank of the card is less or equal to already played card,
                        useless.append((i, j))  # so it can be safely discard
                        if not intentions[j]:
                            intentions[j] = DISCARD
                    if rank < 5 and (color, rank) not in othercards:
                        discardables.append((i, j))  # if the card is not in the pile, it might be discarded
                        if not intentions[j]:
                            intentions[j] = CANDISCARD  # can be discarded (otherwise Intentions is printed as KEEP,
                            # but it actually did not had any effect in original version)

        self.explanation.append(["Intentions"] + map(format_intention, intentions))
        #######################################################################


        ##### 4-2 ) Predict the action of other players given my hint #####
        # TODO: it's nerve wrecking that it keep giving out useless hints (about discarding) when tokens are scarce
        if hints > 0:
            valid = []
            for c in ALL_COLORS:  # giving color hint
                action = (HINT_COLOR, c)
                # reason about expected action of my co-player
                (isvalid,score,expl) = pretend(action, knowledge[1-nr], intentions, hands[1-nr], board)
                self.explanation.append(["Prediction for: Hint Color " + COLORNAMES[c]] + map(format_intention, expl))
                if isvalid:
                    valid.append((action,score))  # all valid (action,score) pair are saved in valid
            
            for r in xrange(5):  # rank hint, analog to color hint
                r += 1
                action = (HINT_NUMBER, r)
                (isvalid,score, expl) = pretend(action, knowledge[1-nr], intentions, hands[1-nr], board)
                self.explanation.append(["Prediction for: Hint Rank " + str(r)] + map(format_intention, expl))
                if isvalid:
                    valid.append((action,score))
                 
            if valid and not result:
                valid.sort(key=lambda (a,s): -s)
                (a,s) = valid[0]  # chose hint with highest score based on heuristics
                if a[0] == HINT_COLOR:
                    result = Action(HINT_COLOR,
                                    pnr=1-nr,  # hint the 1-nr-th (the other player because this is 2-player based)
                                    col=a[1])
                else:
                    result = Action(HINT_NUMBER, pnr=1-nr, num=a[1])
        #############################################################################################


        ##### 5) may discard if nothing else makes sense #####
        # ### A) original, discard a card with lowest expected loss ###
        # if result:
        #     return result
        #
        # self.explanation.append(["My Knowledge"] + map(format_knowledge, knowledge[nr])) # printing
        # possible = [Action(DISCARD, cnr=i) for i in xrange(handsize)]
        #
        # scores = map(lambda p: pretend_discard(p, knowledge[nr], board, trash),  # computed expected loss
        #              possible)  # for each card
        #
        # def format_term((col, rank, n, prob, val)):  # for printing
        #     return COLORNAMES[col] + " " + str(rank) + " (%.2f%%): %.2f" % (prob * 100, val)
        #
        # self.explanation.append(  # for printing
        #     ["Discard Scores"] + map(lambda (a, s, t): "\n".join(map(format_term, t)) + "\n%.2f" % (s), scores))
        # scores.sort(key=lambda (a, s, t): -s)
        # return scores[0][0]  # this will somehow get interpreted as discarding I guess
        # ########################################


        ### B)ADD: If AI can be hinted about KEEP option, here you can just discard the oldest ###
        if result:  # if you any other priorities, return it
            return result

        self.explanation.append(["My Knowledge"] + map(format_knowledge, knowledge[nr]))
        diff = lambda l1, l2: [x for x in l1 if x not in l2]
        maydiscard = diff(xrange(handsize), self.keeplist)  # maydiscard only if not to be kept
        print('may be discarded: ', maydiscard)
        self.keeplist = shift_index(maydiscard[0], self.keeplist) # the oldest from discardable cards will be discarded
        print('shift keeplist after discarding the oldest: ', self.keeplist)
        return Action(DISCARD, cnr=maydiscard[0])
        #############################################################################
        #############################################################################
        #############################################################################


    def inform(self, action, player, game):
        # TODO: does AI take the information of negatively identified cards?
        # e.g.: If I am given a hint that my 1st and 3rd card is red,
        # it should know that 2,4,5 are not red
        if action.type in [PLAY, DISCARD]:
            x = str(action)
            if (action.cnr,player) in self.hints:  # this just never happens?
                print('aaaaaaaaaaaaaa')
                self.hints[(action.cnr,player)] = []
            for i in xrange(10):
                if (action.cnr+i+1,player) in self.hints:
                    print('BBBBBBBBBBBB')  # this neither
                    self.hints[(action.cnr+i,player)] = self.hints[(action.cnr+i+1,player)]
                    self.hints[(action.cnr+i+1,player)] = []
        elif action.pnr == self.pnr:  # when I am hinted
            self.gothint = (action,player)
            self.last_knowledge = game.knowledge[:]
            self.last_board = game.board[:]
            self.last_trash = game.trash[:]
            self.played = game.played[:]
########################################
            

    
def do_sample(knowledge):
    if not knowledge:
        return []
        
    possible = []
    
    for col in ALL_COLORS:
        for i,c in enumerate(knowledge[0][col]):
            for j in xrange(c):
                possible.append((col,i+1))
    if not possible:
        return None
    
    other = do_sample(knowledge[1:])
    if other is None:
        return None
    sample = random.choice(possible)
    return [sample] + other
    
def sample_hand(knowledge):
    result = None
    while result is None:
        result = do_sample(knowledge)
    return result
    
used = {}
for c in ALL_COLORS:
    for i,cnt in enumerate(COUNTS):
        used[(c,i+1)] = 0


class SamplingRecognitionPlayer(Player):
    def __init__(self, name, pnr, other=IntentionalPlayer, maxtime=5000):
        self.name = name
        self.hints = {}
        self.pnr = pnr
        self.gothint = None
        self.last_knowledge = []
        self.last_played = []
        self.last_board = []
        self.other = other
        self.maxtime = maxtime
        self.explanation = []

    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        handsize = len(knowledge[0])
        possible = []

        if self.gothint:
            possiblehands = []
            wrong = 0
            used = {}

            for c in trash + played:
                if c not in used:
                    used[c] = 0
                used[c] += 1

            i = 0
            t0 = time.time()
            while i < self.maxtime:
                i += 1
                h = sample_hand(update_knowledge(knowledge[nr], used))
                newhands = hands[:]
                newhands[nr] = h
                other = self.other("Pinocchio", self.gothint[1])
                act = other.get_action(self.gothint[1], newhands, self.last_knowledge, self.last_trash,
                                       self.last_played, self.last_board, valid_actions, hints + 1)
                lastact = self.gothint[0]
                if act == lastact:
                    possiblehands.append(h)

                    def do(c, i):
                        newhands = hands[:]
                        h1 = h[:]
                        h1[i] = c
                        newhands[nr] = h1
                        print other.get_action(self.gothint[1], newhands, self.last_knowledge, self.last_trash,
                                               self.last_played, self.last_board, valid_actions, hints + 1)
                    # import pdb
                    # pdb.set_trace()
                else:
                    wrong += 1
            # print "sampled", i
            # print len(possiblehands), "would have led to", self.gothint[0], "and not:", wrong
            # print f(possiblehands)
            if possiblehands:
                mostlikely = [(0, 0) for i in xrange(len(possiblehands[0]))]
                for i in xrange(len(possiblehands[0])):
                    counts = {}
                    for h in possiblehands:
                        if h[i] not in counts:
                            counts[h[i]] = 0
                        counts[h[i]] += 1
                    for c in counts:
                        if counts[c] > mostlikely[i][1]:
                            mostlikely[i] = (c, counts[c])
                # print "most likely:", mostlikely
                m = max(mostlikely, key=lambda (card, cnt): cnt)
                second = mostlikely[:]
                second.remove(m)
                m2 = max(second, key=lambda (card, cnt): cnt)
                if m[1] >= m2[1] * a:
                    # print ">>>>>>> deduced!", f(m[0]), m[1],"vs", f(m2[0]), m2[1]
                    knowledge = copy.deepcopy(knowledge)
                    knowledge[nr][mostlikely.index(m)] = iscard(m[0])

        self.gothint = None
        for k in knowledge[nr]:
            possible.append(get_possible(k))

        discards = []
        duplicates = []
        for i, p in enumerate(possible):
            if playable(p, board):
                return Action(PLAY, cnr=i)
            if discardable(p, board):
                discards.append(i)

        if discards:
            return Action(DISCARD, cnr=random.choice(discards))

        playables = []
        for i, h in enumerate(hands):
            if i != nr:
                for j, (col, n) in enumerate(h):
                    if board[col][1] + 1 == n:
                        playables.append((i, j))
        playables.sort(key=lambda (i, j): -hands[i][j][1])
        while playables and hints > 0:
            i, j = playables[0]
            knows_rank = True
            real_color = hands[i][j][0]
            real_rank = hands[i][j][0]
            k = knowledge[i][j]

            hinttype = [HINT_COLOR, HINT_NUMBER]
            if (j, i) not in self.hints:
                self.hints[(j, i)] = []

            for h in self.hints[(j, i)]:
                hinttype.remove(h)

            if HINT_NUMBER in hinttype:
                self.hints[(j, i)].append(HINT_NUMBER)
                return Action(HINT_NUMBER, pnr=i, num=hands[i][j][1])
            if HINT_COLOR in hinttype:
                self.hints[(j, i)].append(HINT_COLOR)
                return Action(HINT_COLOR, pnr=i, col=hands[i][j][0])

            playables = playables[1:]

        for i, k in enumerate(knowledge):
            if i == nr:
                continue
            cards = range(len(k))
            random.shuffle(cards)
            c = cards[0]
            (col, num) = hands[i][c]
            hinttype = [HINT_COLOR, HINT_NUMBER]
            if (c, i) not in self.hints:
                self.hints[(c, i)] = []
            for h in self.hints[(c, i)]:
                hinttype.remove(h)
            if hinttype and hints > 0:
                if random.choice(hinttype) == HINT_COLOR:
                    self.hints[(c, i)].append(HINT_COLOR)
                    return Action(HINT_COLOR, pnr=i, col=col)
                else:
                    self.hints[(c, i)].append(HINT_NUMBER)
                    return Action(HINT_NUMBER, pnr=i, num=num)

        return random.choice([Action(DISCARD, cnr=i) for i in xrange(handsize)])

    def inform(self, action, player, game):
        if action.type in [PLAY, DISCARD]:
            x = str(action)
            if (action.cnr, player) in self.hints:
                self.hints[(action.cnr, player)] = []
            for i in xrange(10):
                if (action.cnr + i + 1, player) in self.hints:
                    self.hints[(action.cnr + i, player)] = self.hints[(action.cnr + i + 1, player)]
                    self.hints[(action.cnr + i + 1, player)] = []
        elif action.pnr == self.pnr:
            self.gothint = (action, player)
            self.last_knowledge = game.knowledge[:]
            self.last_board = game.board[:]
            self.last_trash = game.trash[:]
            self.played = game.played[:]




# I think Eger stopped in middle of the development
class FullyIntentionalPlayer(Player):
    def __init__(self, name, pnr):
        self.name = name
        self.hints = {}
        self.pnr = pnr
        self.gothint = None
        self.last_knowledge = []
        self.last_played = []
        self.last_board = []
    def get_action(self, nr, hands, knowledge, trash, played, board, valid_actions, hints):
        handsize = len(knowledge[0])
        possible = []
        
        ''' ignore hints from other player?'''
        self.gothint = None
        for k in knowledge[nr]:
            possible.append(get_possible(k))
        
        discards = []  # can be discarded for sure
        plays = []  # can be played for sure
        duplicates = []
        for i,card in enumerate(possible):
            if playable(card,board):
                plays.append(i)
            if discardable(card,board):
                discards.append(i)
        #!!!!  I added the line so it can play something
        if plays != []:
            toplay = random.choice(plays)
            print('toplay: ', toplay)
            return Action(PLAY, cnr=toplay)
            
        ''' compute what should be done with cards of teammates'''
        playables = []
        useless = []
        discardables = []
        othercards = trash + board
        intentions = [None for i in xrange(handsize)]
        for i,h in enumerate(hands):
            if i != nr:
                for j,(col,n) in enumerate(h):
                    if board[col][1] + 1 == n:
                        playables.append((i,j))
                        intentions[j] = PLAY
                    if board[col][1] <= n:
                        useless.append((i,j))
                        if not intentions[j]:
                            intentions[j] = DISCARD
                    if n < 5 and (col,n) not in othercards:
                        discardables.append((i,j))
                        if not intentions[j]:
                            intentions[j] = CANDISCARD
        
        
        ''' compute the best hint'''
        if hints > 0:
            valid = []
            for c in ALL_COLORS:
                action = (HINT_COLOR, c)
                #print "HINT", COLORNAMES[c],
                (isvalid,score,expl) = pretend(action, knowledge[1-nr], intentions, hands[1-nr], board)
                #print isvalid, score
                if isvalid:
                    valid.append((action,score))
            
            for r in xrange(5):
                r += 1
                action = (HINT_NUMBER, r)
                #print "HINT", r,
                (isvalid,score,expl) = pretend(action, knowledge[1-nr], intentions, hands[1-nr], board)
                #print isvalid, score
                if isvalid:
                    valid.append((action,score))
            if valid:
                valid.sort(key=lambda (a,s): -s)
                #print valid
                (a,s) = valid[0]
                if a[0] == HINT_COLOR:
                    return Action(HINT_COLOR, pnr=1-nr, col=a[1])
                else:
                    return Action(HINT_NUMBER, pnr=1-nr, num=a[1])
            
        '''NEW: '''
        for i, player in enumerate(knowledge):
            if i == nr or True:  # ???
                continue
            else:
                print('wtf???')
            cards = range(len(player))
            random.shuffle(cards)
            c = cards[0]  # a random card of i-th player
            (col,num) = hands[i][c]  # color and number of that card
            hinttype = [HINT_COLOR, HINT_NUMBER]
            if (c,i) not in self.hints:
                self.hints[(c,i)] = []
            for h in self.hints[(c,i)]:
                hinttype.remove(h)
            if hinttype and hints > 0:
                if random.choice(hinttype) == HINT_COLOR:
                    self.hints[(c,i)].append(HINT_COLOR)
                    return Action(HINT_COLOR, pnr=i, col=col)
                else:
                    self.hints[(c,i)].append(HINT_NUMBER)
                    return Action(HINT_NUMBER, pnr=i, num=num)

        return random.choice([Action(DISCARD, cnr=i) for i in xrange(handsize)])


    def inform(self, action, player, game):
        if action.type in [PLAY, DISCARD]:
            x = str(action)
            if (action.cnr,player) in self.hints:
                self.hints[(action.cnr,player)] = []
            for i in xrange(10):
                if (action.cnr+i+1,player) in self.hints:
                    self.hints[(action.cnr+i,player)] = self.hints[(action.cnr+i+1,player)]
                    self.hints[(action.cnr+i+1,player)] = []
        elif action.pnr == self.pnr:
            self.gothint = (action,player)
            self.last_knowledge = game.knowledge[:]
            self.last_board = game.board[:]
            self.last_trash = game.trash[:]
            self.played = game.played[:]
###########################################################################




def format_card((col,num)):
    return COLORNAMES[col] + " " + str(num)
        
def format_hand(hand):
    return ", ".join(map(format_card, hand))
        

class Game(object):
    def __init__(self, players, log=sys.stdout, format=0):
        self.players = players
        self.hits = 3
        self.hints = 8
        self.current_player = 0
        self.board = map(lambda c: (c,0), ALL_COLORS)
        self.played = []
        self.deck = make_deck()
        self.extra_turns = 0
        self.hands = []
        self.knowledge = []
        self.make_hands()
        self.trash = []
        self.log = log
        self.turn = 1
        self.format = format
        self.dopostsurvey = False
        self.study = False
        if self.format:
            print >> self.log, self.deck
    def make_hands(self):
        handsize = 4
        if len(self.players) < 4:
            handsize = 5
        for i, p in enumerate(self.players):
            self.hands.append([])
            self.knowledge.append([])
            for j in xrange(handsize):
                self.draw_card(i)
    def draw_card(self, pnr=None):
        if pnr is None:
            pnr = self.current_player
        if not self.deck:
            return
        self.hands[pnr].append(self.deck[0])
        self.knowledge[pnr].append(initial_knowledge())
        del self.deck[0]
    def perform(self, action):
        for p in self.players:
            p.inform(action, self.current_player, self)
        if format:
            print >> self.log, "MOVE:", self.current_player, action.type, action.cnr, action.pnr, action.col, action.num
        if action.type == HINT_COLOR:
            self.hints -= 1
            print >>self.log, self.players[self.current_player].name, "hints", self.players[action.pnr].name, "about all their", COLORNAMES[action.col], "cards", "hints remaining:", self.hints
            print >>self.log, self.players[action.pnr].name, "has", format_hand(self.hands[action.pnr])
            for (col,num),knowledge in zip(self.hands[action.pnr],self.knowledge[action.pnr]):
                if col == action.col:
                    for i, k in enumerate(knowledge):
                        if i != col:
                            for i in xrange(len(k)):
                                k[i] = 0
                else:
                    for i in xrange(len(knowledge[action.col])):
                        knowledge[action.col][i] = 0
        elif action.type == HINT_NUMBER:
            self.hints -= 1
            print >>self.log, self.players[self.current_player].name, "hints", self.players[action.pnr].name, "about all their", action.num, "hints remaining:", self.hints
            print >>self.log, self.players[action.pnr].name, "has", format_hand(self.hands[action.pnr])
            for (col,num),knowledge in zip(self.hands[action.pnr],self.knowledge[action.pnr]):
                if num == action.num:
                    for k in knowledge:
                        for i in xrange(len(COUNTS)):
                            if i+1 != num:
                                k[i] = 0
                else:
                    for k in knowledge:
                        k[action.num-1] = 0
        elif action.type == PLAY:
            (col,num) = self.hands[self.current_player][action.cnr]
            print >>self.log, self.players[self.current_player].name, "plays", format_card((col,num)),
            if self.board[col][1] == num-1:
                self.board[col] = (col,num)
                self.played.append((col,num))
                if num == 5:
                    self.hints += 1
                    self.hints = min(self.hints, 8)
                print >>self.log, "successfully! Board is now", format_hand(self.board)
            else:
                self.trash.append((col,num))
                self.hits -= 1
                print >>self.log, "and fails. Board was", format_hand(self.board)
            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self.draw_card()
            print >>self.log, self.players[self.current_player].name, "now has", format_hand(self.hands[self.current_player])
        else:
            self.hints += 1 
            self.hints = min(self.hints, 8)
            self.trash.append(self.hands[self.current_player][action.cnr])
            print >>self.log, self.players[self.current_player].name, "discards", format_card(self.hands[self.current_player][action.cnr])
            print >>self.log, "trash is now", format_hand(self.trash)
            del self.hands[self.current_player][action.cnr]
            del self.knowledge[self.current_player][action.cnr]
            self.draw_card()
            print >>self.log, self.players[self.current_player].name, "now has", format_hand(self.hands[self.current_player])
    def valid_actions(self):
        valid = []
        for i in xrange(len(self.hands[self.current_player])):
            valid.append(Action(PLAY, cnr=i))
            valid.append(Action(DISCARD, cnr=i))
        if self.hints > 0:
            for i, p in enumerate(self.players):
                if i != self.current_player:
                    for col in set(map(lambda (col,num): col, self.hands[i])):
                        valid.append(Action(HINT_COLOR, pnr=i, col=col))
                    for num in set(map(lambda (col,num): num, self.hands[i])):
                        valid.append(Action(HINT_NUMBER, pnr=i, num=num))
        return valid
    def run(self, turns=-1):
        self.turn = 1
        while not self.done() and (turns < 0 or self.turn < turns):
            self.turn += 1
            if not self.deck:
                self.extra_turns += 1
            hands = []
            for i, h in enumerate(self.hands):
                if i == self.current_player:
                    hands.append([])
                else:
                    hands.append(h)
            action = self.players[self.current_player].get_action(self.current_player, hands, self.knowledge, self.trash, self.played, self.board, self.valid_actions(), self.hints)
            self.perform(action)
            self.current_player += 1
            self.current_player %= len(self.players)
        print >>self.log, "Game done, hits left:", self.hits
        points = self.score()
        print >>self.log, "Points:", points
        return points
    def score(self):
        return sum(map(lambda (col,num): num, self.board))
    def single_turn(self):
        if not self.done():
            if not self.deck:
                self.extra_turns += 1
            hands = []
            for i, h in enumerate(self.hands):
                if i == self.current_player:
                    hands.append([])
                else:
                    hands.append(h)
            action = self.players[self.current_player].get_action(self.current_player, hands, self.knowledge, self.trash, self.played, self.board, self.valid_actions(), self.hints)
            self.perform(action)
            self.current_player += 1
            self.current_player %= len(self.players)
    def external_turn(self, action): 
        if not self.done():
            if not self.deck:
                self.extra_turns += 1
            self.perform(action)
            self.current_player += 1
            self.current_player %= len(self.players)
    def done(self):
        if self.extra_turns == len(self.players) or self.hits == 0:
            return True
        for (col,num) in self.board:
            if num != 5:
                return False
        return True
    def finish(self):
        if self.format:
            print >> self.log, "Score", self.score()
            self.log.close()
        
    
class NullStream(object):
    def write(self, *args):
        pass
        
random.seed(123)

playertypes = {"random": Player, "inner": InnerStatePlayer, "outer": OuterStatePlayer, "self": SelfRecognitionPlayer,
               "intentional": IntentionalPlayer, "sample": SamplingRecognitionPlayer, "full": SelfIntentionalPlayer,
               "timed": TimedPlayer}
names = ["Shangdi", "Yu Di", "Tian", "Nu Wa", "Pangu"]
        
        
def make_player(player, i):
    if player in playertypes:
        return playertypes[player](names[i], i)
    elif player.startswith("self("):
        other = player[5:-1]
        return SelfRecognitionPlayer(names[i], i, playertypes[other])
    elif player.startswith("sample("):
        other = player[7:-1]
        if "," in other:
            othername, maxtime = other.split(",")
            othername = othername.strip()
            maxtime = int(maxtime.strip())
            return SamplingRecognitionPlayer(names[i], i, playertypes[othername], maxtime=maxtime)
        return SamplingRecognitionPlayer(names[i], i, playertypes[other])
    return None 
    
def main(args):
    if not args:
        args = ["random"]*3
    if args[0] == "trial":
        treatments = [["intentional", "intentional"], ["intentional", "outer"], ["outer", "outer"]]
        #[["sample(intentional, 50)", "sample(intentional, 50)"], ["sample(intentional, 100)", "sample(intentional, 100)"]] #, ["self(intentional)", "self(intentional)"], ["self", "self"]]
        results = []
        print treatments
        for i in xrange(int(args[1])):
            result = []
            times = []
            avgtimes = []
            print "trial", i+1
            for t in treatments:
                random.seed(i)
                players = []
                for i,player in enumerate(t):
                    players.append(make_player(player,i))
                g = Game(players, NullStream())
                t0 = time.time()
                result.append(g.run())
                times.append(time.time() - t0)
                avgtimes.append(times[-1]*1.0/g.turn)
                print ".",
            print
            print "scores:",result
            print "times:", times
            print "avg times:", avgtimes
        
        return
        
        
    players = []
    
    for i,a in enumerate(args):
        players.append(make_player(a, i))
        
    n = 10000
    out = NullStream()
    if n < 3:
        out = sys.stdout
    pts = []
    for i in xrange(n):
        if (i+1)%100 == 0:
            print "Starting game", i+1
        random.seed(i+1)
        g = Game(players, out)
        try:
            pts.append(g.run())
            if (i+1)%100 == 0:
                print "score", pts[-1]
        except Exception:
            import traceback
            traceback.print_exc()
    if n < 10:
        print pts
    import numpy
    print "average:", numpy.mean(pts)
    print "stddev:", numpy.std(pts, ddof=1)
    print "range", min(pts), max(pts)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])