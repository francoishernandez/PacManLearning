import gzip
import pickle
file = gzip.open("training_data.gz", 'rb')
liste = pickle.load(file)

# Ce script crée la liste des (s, a, r, s') à partir des (s, a, r)
# les actions s'enchainent, donc s' est le s de la ligne suivante
# sauf lorsqu'on est en fin de partie, donc lorsqu'on a une différence 
# de récompense de 500

newlist = []

for i in range(len(liste)) :
    s, a, r = liste[i]
    
    if ( (r > 400) | (r < -400) | (i>=(len(liste)-1)) ) :
        newlist.append((s,a,r,"Final"))
    else :
        sPrime, aPrime , rPrime = liste[i+1]
        newlist.append((s,a,r,sPrime))



file2 = gzip.open("training_data_nextstate.gz", 'w')
pickle.dump(newlist,file2)
file2.close()
