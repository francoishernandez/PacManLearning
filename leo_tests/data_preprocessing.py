import gzip
import pickle
file = gzip.open("training_data.gz", 'rb')
liste = pickle.load(file)

# Ce script crée la liste des (s, a, r, s') et la liste des s à partir 
# des (s, a, r). Les actions s'enchainent, donc s' est le s de la ligne 
# suivante sauf lorsqu'on est en fin de partie, donc lorsqu'on a une  
# différence de récompense de 500

s_a_r_sBis = []
states = []

for i in range(len(liste)) :
    s, a, r = liste[i]
    sCropped = s[0:121,0:195]
    states.append(sCropped)
    
    if ( (r > 400) | (r < -400) | (i>=(len(liste)-1)) ) :
        s_a_r_sBis.append((s,a,r,"Final"))
    else :
        sPrime, _ , _ = liste[i+1]
        sPrimeCropped = sPrime[0:121,0:195]
        s_a_r_sBis.append((s,a,r,sPrimeCropped))



file2 = gzip.open("train_s_a_r_sBis.gz", 'w')
pickle.dump(s_a_r_sBis,file2)
file2.close()

file3 = gzip.open("train_states.gz", 'w')
pickle.dump(states,file3)
file3.close()

