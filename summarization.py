import tensorflow as tf
import tensorflow_hub as hub



embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

print(embeddings)
#
# word = 'test'
#
# sentence = 'this is a test'
#
# #reduce the tf output
# tf.logging.set_verbosity(tf.logging.ERROR)
#
# with tf.Session() as session:
#     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#     message_embeddings = session.run(embed(messages))