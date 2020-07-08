import numpy as np
import spacy
from util import io

nlp = spacy.load('en_core_web_sm')
orig = "E:\PhD\Code\ThesisPipeline\ThesisPipeline\data\processed/runescape/bow\metadata/"

all_dict = np.load(orig + "num_stw.pkl")
freqs = []
words = []
for key, item in all_dict.dfs.items():
    words.append(all_dict.id2token[key])
    freqs.append(item)

freqs = np.asarray(freqs)
words = np.asarray(words)

ids = np.argsort(freqs)[::-1]

freqs = freqs[ids]
words = words[ids]



attack = ["att", "attack"]
strength = ["str", "strength"]
defence = ["def", "defence", "defender", "deffer"]
ranged = ["ranged", "range", "ranging", "ranger"]
prayer = ["prayer", "pray", "praying", "prayer"]
magic = ["magic", "mage", "maging", "mager"]
runecraft = ["runecraft", "rc", "rcing", "runecrafting", "rcer", "runecrafter"]
hitpoints = ["hp", "hitpoints", "hitpoint"]
crafting = ["crafting", "craft", "crafter"]
mining = ["mining", "miner", "mine"]
smithing = ["smithing", "smither", "smith"]
fishing = ["fishing", "fisher", "fish"]
cooking = ["cooking", "cooker", "chef", "cook"]
firemaking = ["fm", "firemaking", "firemaker", "fmer", "firemake"]
woodcutting = ["woodcutting", "yews", "yew", "woodcutter", "wcing", "woodcut"]
agility = ["agil", "agility"]
herblore = ["herb", "herber", "herblore"]
thieving = ["thieve", "thief", "thieving", "thiever"]
fletching = ["fletch", "fletching", "fletcher"]
slayer = ["slayer", "slay"]
construction = ["con", "construction"]
hunter = ["hunter", "hun"]

skills = [attack, strength, defence, ranged, prayer, magic, runecraft, hitpoints, crafting, mining, smithing, fishing, cooking, firemaking, woodcutting, agility, herblore, thieving, fletching, slayer, construction, hunter]


skill_words = []
skill_freqs = []
total_words = []
total_freqs = []
for i in range(len(skills)):
    all_freqs = []
    all_words = []
    for j in range(len(skills[i])):
        for k in range(len(words)):
            if skills[i][j] == words[k]:
                skill_words.append(words[k])
                skill_freqs.append(freqs[k])
                all_freqs.append(freqs[k])
                all_words.append(words[k])
                break
    all_freqs = np.sum(all_freqs)
    all_words = ", ".join(all_words)
    total_words.append(all_freqs)
    total_freqs.append(all_words)


io.write_csv("../../../skills.csv", ["word", "freq"], [skill_words, skill_freqs], key=list(range(len(skill_words))))
io.write_csv("../../../all_skills.csv", ["word", "freq"], [total_words, total_freqs], key=list(range(len(total_words))))



drew = []
false = []

drew_f = []
false_f = []
for i in range(len(words)):
    if str(words[i].astype('U')) in nlp.vocab:
        drew.append(words[i])
        drew_f.append(freqs[i])
    else:
        false.append(words[i])
        false_f.append(freqs[i])


all_words_another_vocab = np.load("E:\PhD\Code\ThesisPipeline\ThesisPipeline\data\processed/newsgroups/bow\metadata/num_stw_all_words_2.npy")

in_other_words = []
in_other_freqs = []

for i in range(len(false)):
    if false[i] not in all_words_another_vocab:
        in_other_words.append(false[i])
        in_other_freqs.append(false_f[i])

io.write_csv("../../../freqs.csv", ["word", "freq"], [in_other_words, in_other_freqs], key=list(range(len(in_other_words))))

interest_array = [1,5,10,14,15,16,17,23,28,29,30,31,32,36,39,45,46,49,53,52,58,59,60,61,62,63,65,64,68,69,70,71,74,78,79,82,83,84,85,86,88,87,93,94,95,96,97,98,99,100,101,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,128,129,130,132,133,134,135,136,137,142,143,144,141,145,146,147,149,150,151,152,153,157,158,163,164,165,166,167,171,173,174,176,178,179,180,181,182,183,185,186,188,189,190,192,193,195,198,204,205,208,210,211,213,214,217,227,229,232,239,242,247,251,253,255,258,258,269,271,276]
words_of_interest = np.asarray(in_other_words)[interest_array]
freqs_of_interest = np.asarray(in_other_freqs)[interest_array]

io.write_csv("../../../interest.csv", ["word", "freq"], [words_of_interest, freqs_of_interest], key=list(range(len(words_of_interest))))


print("ok")