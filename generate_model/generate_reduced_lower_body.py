from matplotlib import pyplot as plt
import numpy as np
from biorbd.model_creation import (
    Axis,
    BiomechanicalModel,
    BiomechanicalModelReal,
    SegmentCoordinateSystem,
    InertiaParameters,
    Mesh,
    Segment,
    Marker,
    Translations,
    Rotations,
    C3dData,
)
import biorbd


def chord_function(offset, known_center_of_rotation, center_of_rotation_marker, plane_marker, direction: int = 1):
    n_frames = offset.shape[0]

    # Create a coordinate system from the markers
    axis1 = plane_marker[:3, :] - known_center_of_rotation[:3, :]
    axis2 = center_of_rotation_marker[:3, :] - known_center_of_rotation[:3, :]
    axis3 = np.cross(axis1, axis2, axis=0)
    axis1 = np.cross(axis2, axis3, axis=0)
    axis1 /= np.linalg.norm(axis1, axis=0)
    axis2 /= np.linalg.norm(axis2, axis=0)
    axis3 /= np.linalg.norm(axis3, axis=0)
    rt = np.identity(4)
    rt = np.repeat(rt, n_frames, axis=1).reshape((4, 4, n_frames))
    rt[:3, 0, :] = axis1
    rt[:3, 1, :] = axis2
    rt[:3, 2, :] = axis3
    rt[:3, 3, :] = known_center_of_rotation[:3, :]

    # The point of interest is the chord from center_of_rotation_marker that has length 'offset' assuming
    # the diameter is the distance between center_of_rotation_marker and known_center_of_rotation.
    # To compute this, project in the rt knowing that by construction, known_center_of_rotation is at 0, 0, 0
    # and center_of_rotation_marker is at a diameter length on y
    diameter = np.linalg.norm(known_center_of_rotation[:3, :] - center_of_rotation_marker[:3, :], axis=0)
    x = offset * direction * np.sqrt(diameter**2 - offset**2) / diameter
    y = (diameter**2 - offset**2) / diameter

    # project the computed point in the global reference frame
    vect = np.concatenate((x[np.newaxis, :], y[np.newaxis, :], np.zeros((1, n_frames)), np.ones((1, n_frames))))

    def rt_times_vect(m1, m2):
        return np.einsum("ijk,jk->ik", m1, m2)

    return rt_times_vect(rt, vect)


def point_on_vector(coef: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Computes the 3d position of a point using this equation: start + coef * (end - start)

    Parameters
    ----------
    coef
        The coefficient of the length of the segment to use. It is given from the starting point
    start
        The starting point of the segment
    end
        The end point of the segment

    Returns
    -------
    The 3d position of the point
    """

    return start + coef * (end - start)


def project_point_on_line(start_line: np.ndarray, end_line: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Project a point on a line defined by to points (start_line and end_line)

    Parameters
    ----------
    start_line
        The starting point of the line
    end_line
        The ending point of the line
    point
        The point to project

    Returns
    -------
    The projected point
    -------

    """

    def dot(v1, v2):
        return np.einsum("ij,ij->j", v1, v2)

    sp = (point - start_line)[:3, :]
    line = (end_line - start_line)[:3, :]
    return start_line[:3, :] + dot(sp, line) / dot(line, line) * line


class ReducedModel(BiomechanicalModel):

    def __init__(
        self,
        body_mass: float,
    ):
        super(ReducedModel, self).__init__()
        self.body_mass = body_mass
        self._define_kinematic_model()

    def _define_kinematic_model(self):
        self["Ground"] = Segment()

        self["Pelvis"] = Segment(
            parent_name="Ground",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=self._pelvis_joint_center,
                first_axis=Axis(name=Axis.Name.X, start=lambda m, bio: (m["LPSIS"] + m["RPSIS"]) / 2, end="RASIS"),
                second_axis=Axis(name=Axis.Name.Y, start="RASIS", end="LASIS"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("LPSIS", "RPSIS", "RASIS", "LASIS", "LPSIS")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.6 * self.body_mass,  # TODO: change
                center_of_mass=self._pelvis_center_of_mass,  # TODO: change
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.145 * self.body_mass,  # TODO: change
                    coef=(0.31, 0.31, 0.3),  # TODO: change
                    start=self._pelvis_joint_center(m, bio),  # TODO: change
                    end=self._pelvis_center_of_mass(m, bio),  # TODO: change
                ),
            ),
        )
        self["Pelvis"].add_marker(Marker("LPSIS", is_technical=True, is_anatomical=True))
        self["Pelvis"].add_marker(Marker("RPSIS", is_technical=True, is_anatomical=True))
        self["Pelvis"].add_marker(Marker("LASIS", is_technical=True, is_anatomical=True))
        self["Pelvis"].add_marker(Marker("RASIS", is_technical=True, is_anatomical=True))
        self["Pelvis"].add_marker(Marker("RA", is_technical=True, is_anatomical=True))
        self["Pelvis"].add_marker(Marker("LA", is_technical=True, is_anatomical=True))

        self["RFemur"] = Segment(
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._hip_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._hip_joint_center(m, bio, "R"),
                ),
                second_axis=self._knee_axis("R"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._hip_joint_center(m, bio, "R"),
                    lambda m, bio: self._knee_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio:  0.142 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.3612, start=self._hip_joint_center(m, bio, "R"), end=self._knee_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.142 * self.body_mass,
                    coef=(0.32, 0.32, 0.16),
                    start=self._hip_joint_center(m, bio, "R"),
                    end=self._knee_joint_center(m, bio, "R"),
                ),
            ),
        )
        self["RFemur"].add_marker(Marker("RLFE", is_technical=True, is_anatomical=True))
        self["RFemur"].add_marker(Marker("RMFE", is_technical=True, is_anatomical=True))

        self["RTibia"] = Segment(
            parent_name="RFemur",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._ankle_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                ),
                second_axis=self._knee_axis("R"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._knee_joint_center(m, bio, "R"),
                    lambda m, bio: self._ankle_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0433 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.4416, start=self._knee_joint_center(m, bio, "R"), end=self._ankle_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0433 * self.body_mass,
                    coef=(0.3, 0.3, 0.2),
                    start=self._knee_joint_center(m, bio, "R"),
                    end=self._ankle_joint_center(m, bio, "R"),
                ),
            ),
        )
        self["RTibia"].add_marker(Marker("RLM", is_technical=True, is_anatomical=True))
        self["RTibia"].add_marker(Marker("RSPH", is_technical=True, is_anatomical=True))

        self["RFoot"] = Segment(
            parent_name="RTibia",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._ankle_joint_center(m, bio, "R"),
                first_axis=Axis(Axis.Name.Y, start="RLM", end="RSPH"),
                second_axis=Axis(Axis.Name.X, start="RLM", end="RTT2"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("RLM", "RTT2", "RSPH")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0133 * self.body_mass,  # TODO: CoM
                center_of_mass=lambda m, bio: np.array([0, 0, 0, 1]),
                inertia=lambda m, bio: [0, 0, 0],
            ),
        )
        self["RFoot"].add_marker(Marker("RTT2", is_technical=True, is_anatomical=True))

        self["LFemur"] = Segment(
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._hip_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._hip_joint_center(m, bio, "L"),
                ),
                second_axis=self._knee_axis("L"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._hip_joint_center(m, bio, "L"),
                    lambda m, bio: self._knee_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.142 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.3612, start=self._hip_joint_center(m, bio, "L"), end=self._knee_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.142 * self.body_mass,
                    coef=(0.32, 0.32, 0.16),
                    start=self._hip_joint_center(m, bio, "L"),
                    end=self._knee_joint_center(m, bio, "L"),
                ),
            ),
        )
        self["LFemur"].add_marker(Marker("LLFE", is_technical=True, is_anatomical=True))
        self["LFemur"].add_marker(Marker("LMFE", is_technical=True, is_anatomical=True))

        self["LTibia"] = Segment(
            parent_name="LFemur",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._ankle_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                ),
                second_axis=self._knee_axis("L"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._knee_joint_center(m, bio, "L"),
                    lambda m, bio: self._ankle_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0433 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.5, start=self._knee_joint_center(m, bio, "L"), end=self._ankle_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0433 * self.body_mass,
                    coef=(0.3, 0.3, 0.2),
                    start=self._knee_joint_center(m, bio, "L"),
                    end=self._ankle_joint_center(m, bio, "L"),
                ),
            ),
        )
        self["LTibia"].add_marker(Marker("LLM", is_technical=True, is_anatomical=True))
        self["LTibia"].add_marker(Marker("LSPH", is_technical=True, is_anatomical=True))

        self["LFoot"] = Segment(
            parent_name="LTibia",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._ankle_joint_center(m, bio, "L"),
                first_axis=Axis(Axis.Name.Y, start="LSPH", end="LLM"),
                second_axis=Axis(Axis.Name.X, start="LLM", end="LTT2"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("LLM", "LTT2", "LSPH")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0133 * self.body_mass,
                center_of_mass=lambda m, bio: np.array([0, 0, 0, 1]),
                inertia=lambda m, bio: [0, 0, 0],
            ),
        )
        self["LFoot"].add_marker(Marker("LTT2", is_technical=True, is_anatomical=True))

    def _lumbar_5(self, m, bio):
        right_hip = self._hip_joint_center(m, bio, "R")
        left_hip = self._hip_joint_center(m, bio, "L")
        return np.nanmean((left_hip, right_hip), axis=0) + np.array((0.0, 0.0, 0.828, 0))[:, np.newaxis] * np.repeat(
            np.linalg.norm(left_hip - right_hip, axis=0)[np.newaxis, :], 4, axis=0
        )

    def _pelvis_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        return (m["LPSIS"] + m["RPSIS"] + m["LASIS"] + m["RASIS"]) / 4

    def _pelvis_center_of_mass(self, m: dict, bio: BiomechanicalModelReal) -> np.ndarray:
        """
        This computes the center of mass of the thorax

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        """
        right_hip = self._hip_joint_center(m, bio, "R")
        left_hip = self._hip_joint_center(m, bio, "L")
        p = self._pelvis_joint_center(m, bio)  # Make sur the center of mass is symmetric
        p[2, :] += 0.925 * (self._lumbar_5(m, bio) - np.nanmean((left_hip, right_hip), axis=0))[2, :]
        return p

    def _thorax_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        return m["SUP"]

    def _legs_length(self, m, bio: BiomechanicalModelReal):
        # TODO: Verify 95% makes sense
        return {
            "R": np.nanmean(np.linalg.norm(m["RASIS"][:3, :]-m["RLM"][:3, :], axis=0)),
            "L": np.nanmean(np.linalg.norm(m["LASIS"][:3, :]-m["LLM"][:3, :], axis=0)),
        }

    def _hip_joint_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the hip joint center. The LegLength is not provided, the height of the TROC is used (therefore assuming
        the subject is standing upright during the static trial)

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """
        """
               inter_asis = np.nanmean(np.linalg.norm(m["LASIS"][:3, :] - m["RASIS"][:3, :], axis=0))
               legs_length = self._legs_length(m, bio)
               mean_legs_length = np.nanmean((legs_length["R"], legs_length["L"]))
               asis_troc_dist = 0.1288 * legs_length[side] - 0.04856

               c = mean_legs_length * 0.115 - 0.0153
               aa = inter_asis / 2
               theta = 0.5
               beta = 0.314
               x = c * np.cos(theta) * np.sin(beta) - asis_troc_dist * np.cos(beta)
               y = -(c * np.sin(theta) - aa)
               z = -c * np.cos(theta) * np.cos(beta) - asis_troc_dist * np.sin(beta)
               return m[f"{side}ASIS"] + np.array((x, y, z, 0))[:, np.newaxis]

        """
        inter_asis = np.nanmean(np.linalg.norm(m["LASIS"][:3, :] - m["RASIS"][:3, :], axis=0))
        legs_length = self._legs_length(m, bio)
        PJC = self._pelvis_joint_center(m, bio)

        mean_legs_length = np.nanmean((legs_length["R"], legs_length["L"]))
        asis_troc_dist = 0.1288 * legs_length[side] - 0.04856
        #asis_troc_dist = np.nanmean(np.linalg.norm(m["RGT"][:3, :] - m["RASIS"][:3, :], axis=0))
        x = 0.011-0.063 * mean_legs_length
        y = 8/1000 + 0.086 * mean_legs_length
        z = -9/1000 - 0.078 * mean_legs_length
        Axe = m[f"{side}ASIS"]-PJC
        dir = np.mean(Axe[1,:])/np.abs(np.mean(Axe[1,:]))
        x = PJC[0,:] - x
        y = PJC[1,:] + y*dir
        z = PJC[2,:] + z
        return np.array((x, y, z, m[f"{side}ASIS"][3,:])) #m[f"{side}ASIS"] + (np.array((x, y, z, 0))[:, np.newaxis]/2)




    def _knee_axis(self, side) -> Axis:
        """
        Define the knee axis

        Parameters
        ----------
        side
            If the markers are from the right ("R") or left ("L") side
        """
        if side == "R":
            return Axis(Axis.Name.Y, start=f"{side}LFE", end=f"{side}MFE")
        elif side == "L":
            return Axis(Axis.Name.Y, start=f"{side}MFE", end=f"{side}LFE")
        else:
            raise ValueError("side should be 'R' or 'L'")

    def _knee_joint_center(self, m, bio: BiomechanicalModelReal, side) -> np.ndarray:
        """
        Compute the knee joint center. This is a simplified version since the KNM exists

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """
        return (m[f"{side}MFE"] + m[f"{side}LFE"]) / 2

    def _ankle_joint_center(self, m, bio: BiomechanicalModelReal, side) -> np.ndarray:
        """
        Compute the ankle joint center. This is a simplified version sie ANKM exists

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        return (m[f"{side}SPH"] + m[f"{side}LM"]) / 2

    @property
    def dof_index(self) -> dict[str, tuple[int, ...]]:
        """
        Returns a dictionary with all the dof to export to the C3D and their corresponding XYZ values in the generalized
        coordinate vector
        """

        # TODO: Some of these values as just copy of their relative
        return {"LHip": (15, 16, 17),
                "LKnee": (18, 19, 20),
                "LAnkle": (21, 22, 23),
                "LAbsAnkle": (24, 25, 26),
                "RHip": (6, 7, 8),
                "RKnee": (9, 10, 11),
                "RAnkle": (12, 13, 14),
                "RAbsAnkle": (33, 34, 35),
                "LPelvis": (3, 4, 5),
                "RPelvis": (3, 4, 5),
                }


    def personalize_model(self, static_trial: str, model_path: str = "temporary.bioMod"):
        """
        Collapse the generic model according to the data of the static trial

        Parameters
        ----------
        static_trial
            The path of the c3d file of the static trial to create the model from
        model_path
            The path of the generated bioMod file
        """

        self.write(save_path=model_path, data=C3dData(static_trial))
        self.model = biorbd.Model(model_path)


def main():

    static_trial = "../example/ECH_static.c3d"

    # Generate the personalized kinematic model
    tools = ReducedModel(body_mass=66)
    tools.personalize_model(static_trial)



if __name__ == "__main__":
    main()
