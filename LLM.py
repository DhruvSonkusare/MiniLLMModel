from keras.layers import TextVectorization,Embedding,LSTM,Dense
from keras.models import Sequential
import tensorflow as tf




corpus = [
    "A stray dog shivered in a cold alleyway", 
    "His fur was matted with dirt and rain", 
    "Hunger gnawed at his empty belly", 
    "Cars rushed by on the wet city street", 
    "People walked past without a second glance", 
    "He sniffed the air for any sign of food", 
    "A warm scent drifted from a nearby bakery", 
    "The dog crept closer with hopeful eyes", 
    "The bakery door opened and a bell jingled", 
    "A kind old woman stepped out with a tray of scraps", 
    "She glanced at the dog but stayed the course to her task", 
    "Her name was Mrs. Chen, owner of the bakery", 
    "She placed a small plate on the curb and moved away", 
    "The dog hesitated, then tasted the offered morsel",
    "His tail gave a tentative wag as the bite disappeared", 
    "He looked up, studying the bakery door for permission", 
    "A young boy named Theo appeared from an alley", 
    "Theo watched the dog with wide, curious eyes", 
    "Mrs. Chen winked at Theo and whispered a plan", 
    "The dog rose slowly, savoring the small kindness", 
    "The boy stepped closer, speaking softly to ease fear", 
    "Nearby, a stray cat observed from a safe distance", 
    "The cat hissed, prompting the dog to drop the guard", 
    "Theo offered a friendly nod and extended a hand", 
    "The dog sniffed, then nudged Theo's palm with his nose",
    "Theo laughed softly and touched the back of the dog’s ear", 
    "Mrs. Chen returned with a bowl of water", 
    "She poured fresh water into the bowl and set it down", 
    "Theo fetched a worn collar from his backpack", 
    "He asked Mrs. Chen for permission to adopt the dog together", 
    "She smiled, impressed by their teamwork and care", 
    "They walked to a small shelter at the edge of the lot", 
    "The shelter keeper, Mr. Ruiz, welcomed them warmly", 
    "Mr. Ruiz agreed to help with medical checks and vaccinations", 
    "On the way back, Theo held the leash loosely, guiding gently", 
    "The dog moved with cautious steps but trust began to rise", 
    "They reached home and the door opened to a bright kitchen", 
    "The dog sniffed the air, discovering comforting smells", 
    "Theo’s mother grinned at the sight of the new friend", 
    "She offered a soft blanket and a cozy corner by the window", 
    "The dog curled up, content, eyes drifting closed", 
    "The house settled into a quiet, hopeful rhythm", 
    "Outside, the park sounds drifted through the open window", 
    "Two new companions shared a silent moment of belonging", 
    "The dog dreamed of gentle streets and warm hands", 
    "The girl, Lily, whispered a promise to care and protect", 
    "And in that promise, two hearts learned what family means"
]


input_texts = []
target_words = [] 

for line in corpus:
    words = line.lower().split() 
    for i in range(1, len(words)):
        input_texts.append(" ".join(words[:i])) 
        target_words.append(words[i])


print(input_texts)
print(target_words)

vectorizer=TextVectorization(output_sequence_length=10)
vectorizer.adapt(corpus)

vocabulary=vectorizer.get_vocabulary()
print("Vocabulary:",vocabulary)
print("Length of Vocabulary:",len(vocabulary))

y=[]
for word in target_words:
    encoded=vectorizer([word]).numpy()
    value=encoded[0][0]
    y.append(value)
print(target_words,y)
model=Sequential([

    vectorizer,
    Embedding(input_dim=len(vocabulary),output_dim=50),

    LSTM(32),
    Dense(len(vocabulary),activation="sigmoid")

])


model.compile(loss="sparse_categorical_crossentropy",metrics=['accuracy'],optimizer="adam")

model.fit(tf.constant(input_texts),tf.constant(y),epochs=1000)
predicted_word=''
test='stray dog'
for i in range(10):
    

    result=model.predict(tf.constant([test]))

    print("The value at:",result.argmax())

    predicted_word=vocabulary[result.argmax()]

    print(predicted_word)
    test=test+" "+predicted_word

print(test)





