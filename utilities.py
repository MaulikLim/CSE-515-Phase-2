import numpy as np

def extract_type_weight_pairs(labels, metrics):
  type_metrics = {}
  for x in range(len(labels)):
    image_type = labels[x].split("-")[1]
    type_data = []
    if image_type in type_metrics:
      type_data = type_metrics[image_type]
    type_data.append(metrics[x])
    type_metrics[image_type] = type_data
  y = metrics.shape[1]
  type_weights = []
  types = []
  index = 0
  for image_type, data in type_metrics.items():
    types.append(image_type)
    count = 0
    type_weight = np.zeros(y)
    for d in data:
      type_weight += d
      count += 1
    type_weights.append(type_weight/count)
    index += 1
  return [types, type_weights]

def print_semantics_type(labels, metrics):
    subjects, subject_weights = extract_type_weight_pairs(labels, metrics)
    subject_weights = np.array(subject_weights)
    for x in range(subject_weights.shape[1]):
        print("latent semantic "+str(x)+":", end=" ")
        semantic_weights = subject_weights[:,x:x+1].flatten()
        sorted_order = np.flip(np.argsort(semantic_weights))
        for x in sorted_order:
            print(subjects[x]+"="+str(semantic_weights[x]), end=" ")
        print()

def extract_subject_weight_pairs(labels, metrics):
  subject_metrics = {}
  for x in range(len(labels)):
    subject = labels[x].split("-")[2]
    subject_data = []
    if subject in subject_metrics:
      subject_data = subject_metrics[subject]
    subject_data.append(metrics[x])
    subject_metrics[subject] = subject_data
  y = metrics.shape[1]
  subject_weights = []
  subjects = []
  index = 0
  for sub, data in subject_metrics.items():
    subjects.append(sub)
    count = 0
    subject_weight = np.zeros(y)
    for d in data:
      subject_weight += d
      count += 1
    subject_weights.append(subject_weight/count)
    index += 1
  return [subjects, subject_weights]

def print_semantics_sub(labels, metrics):
    subjects, subject_weights = extract_subject_weight_pairs(labels, metrics)
    subject_weights = np.array(subject_weights)
    for x in range(subject_weights.shape[1]):
        print("latent semantic "+str(x)+":", end=" ")
        semantic_weights = subject_weights[:,x:x+1].flatten()
        sorted_order = np.flip(np.argsort(semantic_weights))
        for x in sorted_order:
            print(subjects[x]+"="+str(semantic_weights[x]), end=" ")
        print()
