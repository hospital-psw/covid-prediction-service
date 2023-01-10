import json
import numpy as np


class Symptoms:
    def __init__(
        self, breathing_problems: bool, fever: bool, dry_cough: bool, sore_throat: bool,
        running_nose: bool, asthma: bool, chronic_lung_disease: bool, headache: bool,
        diabetes: bool, hyper_tension: bool, fatigue: bool, gastrointestinal: bool,
        abroad_travel: bool, contact_with_covid: bool, attended_large_gathering: bool,
        visited_public_exposed_places: bool, family_working_in_public_exposed_places: bool
        ):

        self.breathing_problems = breathing_problems
        self.fever = fever
        self.dry_cough = dry_cough
        self.sore_throat = sore_throat
        self.running_nose = running_nose
        self.asthma = asthma
        self.chronic_lung_disease = chronic_lung_disease
        self.headache = headache
        self.diabetes = diabetes
        self.hyper_tension = hyper_tension
        self.fatigue = fatigue
        self.gastrointestinal = gastrointestinal
        self.abroad_travel = abroad_travel
        self.contact_with_covid = contact_with_covid
        self.attended_large_gathering = attended_large_gathering
        self.visited_public_exposed_places = visited_public_exposed_places
        self.family_working_in_public_exposed_places = family_working_in_public_exposed_places

    def to_numpy(self) -> np.ndarray:
        return np.array([*self.__dict__.values()], dtype=int)

    @staticmethod
    def load_from_json_str(json_str: str):
        json_dict = json.loads(json_str)
        symptoms = Symptoms(*[True for val in json_dict.values])    # init empty object 

        # for each attr name set it with the value in the json
        for key in symptoms.__dict__.keys():
            setattr(symptoms, key, json_dict[key])

        return symptoms