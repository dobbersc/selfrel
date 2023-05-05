# ray.init()
#
# sentence = Sentence("Berlin is the capital of Germany.")
# sentence2 = Sentence("Berlin is the capital of Germany.")
# model_ = Classifier.load("flair/ner-english-large")
#
# model_.predict(sentence2)
#
# model_ref = ray.put(model_)
#
# a = Predictor.options(num_gpus=1).remote(model_ref, index=2)
# new = ray.get(a.predict.remote(sentence))
#
# print(sentence2)
# print(new)
