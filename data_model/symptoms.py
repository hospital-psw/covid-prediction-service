import numpy as np
from typing import Dict


class Symptoms:
    def __init__(
        self, breathing_problems: bool, fever: bool, dry_cough: bool, sore_throat: bool,
        running_nose: bool, asthma: bool, chronic_lung_disease: bool, headache: bool,
        heart_disease: bool, diabetes: bool, hyper_tension: bool, fatigue: bool,
        gastrointestinal: bool, abroad_travel: bool, contact_with_covid: bool,
        attended_large_gathering: bool, visited_public_exposed_places: bool,
        family_working_in_public_exposed_places: bool
        ):

        self.breathing_problems: bool = breathing_problems
        self.fever: bool = fever
        self.dry_cough: bool = dry_cough
        self.sore_throat: bool = sore_throat
        self.running_nose: bool = running_nose
        self.asthma: bool = asthma
        self.chronic_lung_disease: bool = chronic_lung_disease
        self.headache: bool = headache
        self.heart_disease: bool = heart_disease
        self.diabetes: bool = diabetes
        self.hyper_tension: bool = hyper_tension
        self.fatigue: bool = fatigue
        self.gastrointestinal: bool = gastrointestinal
        self.abroad_travel: bool = abroad_travel
        self.contact_with_covid: bool = contact_with_covid
        self.attended_large_gathering: bool = attended_large_gathering
        self.visited_public_exposed_places: bool = visited_public_exposed_places
        self.family_working_in_public_exposed_places: bool = family_working_in_public_exposed_places

    def to_numpy(self) -> np.ndarray:
        return np.array([*self.__dict__.values()], dtype=int).reshape(1, -1)

    @staticmethod
    def load_from_json_dict(json_dict: Dict[str, bool]):
        # init empty object 
        symptoms = Symptoms(*[True for val in json_dict.values()])

        # for each attr name set it with the value in the json
        for key in symptoms.__dict__.keys():
            setattr(symptoms, key, json_dict[key])

        return symptoms