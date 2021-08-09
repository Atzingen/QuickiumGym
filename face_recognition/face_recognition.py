from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import numpy as np
import os
import cv2
from copy import deepcopy


def get_img_array_formated(image_path, resize=(224, 224)):
    """Get img array from file path, resize, convert color to Blue, Green and Red

    Args:
        image_path (string): file path of image
        resize (tuple): width and height of final image

    Returns:
        np.array: return a np.array containing image pixels value.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, resize)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def get_template_faces(template_path):
    """Get true face list from folder path and a dict with reference
    Args:
        template_path (string): folder path containing other folders with
        people names, and their folder must contain the true faces.

    Returns:
        tuple: template_faces containing images array of those faces, and a
        reference dict that keys are people names and values are list
        with indixes of template_faces list
    """
    template_faces = []
    reference = {}
    count = 0
    for name in os.listdir(template_path):
        reference[name] = []
        for img in os.listdir(f'{template_path}/{name}'):
            img_array = get_img_array_formated(f'{template_path}/{name}/{img}')
            template_faces.append(img_array)
            reference[name].append(count)
            count += 1
    return (template_faces, reference)


def calculate_match_scores(face_array,
                           template_faces,
                           reference,
                           model,
                           metric='max'):
    """This calculate the match scores between a face and other template faces.
       It compare the face got by face_array and compare
       with other faces got by template_faces.
       The template faces can have multiple images, the final score can be a
       mean of the comparasion or the image most semelhante

    Args:
        face_array (string): image to be compared
        template_faces (list): list contaning imgs array of people
        reference ([string]): dicts that keys are people name and values index
        of their face metric (str, optional): method to use to compare.
        Defaults to 'max'.
        model (keras.model): model to calculate score of images

    Returns:
        dict: dict that keys are people name and values the porcent of match
    """
    known_face = face_array
    data = template_faces
    data.append(known_face)
    samples = np.asarray(data, 'float32')
    samples = preprocess_input(samples, version=2)
    model_scores = model.predict(samples)
    scores = [cosine(model_scores[-1], i) for i in model_scores]
    scores = [(100 - (100 * i)) for i in scores]
    scores_array = np.array(scores)
    for name in reference:
        if metric == 'max':
            reference[name] = scores_array[reference[name]].max()
        elif metric == 'mean':
            reference[name] = scores_array[reference[name]].mean()
    return reference


def get_people_scores(face_array,
                      template_path,
                      model,
                      just_max=False,
                      threshold=0.0):
    """Get person's name of a a face image. This function require
       the functions get_template_faces and calculate_match_scores

    Args:
        face_array (string): filepath of image to be compared
        template_path (string): folder path containing other folders with
        people names, and their folder must contain the true faces.
        just_max (Bool): True to return just one person name, False to return
        all people name with respectives scores
        threshold (float): return just people with score percent
        greater than threshold.

    Returns:
        string: Person name
    """
    template_faces, reference = get_template_faces(template_path)
    scores_dict = calculate_match_scores(
        face_array,
        template_faces,
        reference,
        model)

    people_name = deepcopy(scores_dict).keys()
    if threshold:
        for key in people_name:
            if scores_dict[key] < threshold:
                del scores_dict[key]
    if just_max:
        person_name = max(scores_dict, key=scores_dict.get)
        return person_name
    return scores_dict


def compare_two_faces(face_array_a,
                      face_array_b,
                      model):
    """Compare two image faces and return if both are the same person

    Args:
        face_array_a (np.array): face_array_a
        face_array_b (np.array): face_array_b
        model (keras.model): VGGface model

    Returns:
        float: percent of match person
    """
    face_array_a = cv2.resize(face_array_a, (224, 224))
    face_array_a = cv2.cvtColor(face_array_a, cv2.COLOR_RGB2BGR)
    face_array_b = cv2.resize(face_array_b, (224, 224))
    face_array_b = cv2.cvtColor(face_array_b, cv2.COLOR_RGB2BGR)
    samples = np.asarray([face_array_a, face_array_b], 'float32')
    samples = preprocess_input(samples, version=2)
    model_scores = model.predict(samples)
    score = cosine(model_scores[0], model_scores[1])
    score = (100 - (100 * score))
    return score


if __name__ == '__main__':
    face_path = 'frames/henrique/2/frame_103_[8.73100918].jpg'
    template_path = 'gabarito'
    model = VGGFace(
        model='resnet50',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg')
    get_people_scores(
        get_img_array_formated('frame_1_[8.93581806].jpg'),
        template_path='true_faces', model=model)
