from nltk.corpus import wordnet

def lesk_algorithm_unit(context_sentence, ambiguous_word, pos = None, synset = None):
	final_context = set(context_sentence) #Το κάνω set έτσι ώστε να μπορώ να χρησιμοποιήσω την εντολή intersection των set που μου επιστρέφει τα κοινά στοιχεία δύο sets έτσι ώστε να μπορώ να βρώ εύκολα τους κοινούς όρους ανάμεσα στα definitions τους
	if synset is None:
		synset = wordnet.synsets(ambiguous_word) #βρίσκω τα set της λέξης της οποίας προσπαθώ να καθορίσω την σημασία της
		
	if pos :
		synset = [a for a in synset if str(a.pos()) == pos] #Δίνω την δυνατότητα να επιλέξει κανείς το σύνολο που με την κατάλληλη σημασία που θα επιστρέφει να αποτελείται απο στοιχεία κάποιου συγκεκριμένου μέρους του λόγου
	
	if not synset:
		return None # Σε αυτό το σημείο εάν το wordnet δεν περιέχει σύνολα με παρόμοιας σημασίας λέξης να επιστρέψει None. Αυτό συμβαίνει σε λέξει όπως for, to, if, why e.t.c.
	
	help_list = []
	for a in synset:
		sa = (len(final_context.intersection(set(a.definition().split()))), a) #Δηλώνω για κάθε sunset του ambiguous_word μία tuple με δύο στοιχεία. Το πρώτο είναι το πόσες λέξεις υπάρχουν κοινές στο definition του συγκεκριμένου synset(που είναι το δεύτερο στοιχείο) και στην context_sentence (που θεωρώ ότι είναι η γειτωνιά αυτής της λέξεις). 
		help_list.append(sa)  #Προσθέτω τα tuples σε μία list
	
	sense = max(help_list) #Xρησιμοποιώ την εντολή max προκειμένου να πάρω την tuple με το μεγαλύτερο πρώτο όρισμα (αυτό σημαίνει ότι θα πάρω την tuple, τις οποίας το δεύτερο όρισμα θα είναι το synset που ο ορισμός του θα έχει τους περισσότερους κοινούς όρους με την γειτωνια την λέξης της οποίας την σημασία ψάχνω)
	
	return sense[1] #Επιστρέφω το synset
	

# m = lesk_algorithm_unit(['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.'], 'bank', 'n')
# print(m)

