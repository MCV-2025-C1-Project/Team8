from enum import Enum
from keypoint_descriptors.orb_descriptor import orb_descriptor


class KeypointDescriptorMethod(Enum):
    ORB = "orb"


    def compute(
        self, 
        img,
    ):
        """Compute the local descriptors for the given image with optional preprocessing."""
        if self == KeypointDescriptorMethod.ORB:
            return orb_descriptor(img, nfeatures=500)
    
    @property
    def name(self) -> str:
        return self.value.upper()