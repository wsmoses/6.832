# -*- coding: utf8 -*-

import argparse
import math
import os.path
import time

import numpy as np

from pydrake.all import (VectorSystem)

class Hopper2dController(VectorSystem):
    def __init__(self, hopper, 
        desired_lateral_velocity = 0.0,
        print_period = 0.0):
        '''
        Controls a planar hopper described
        in raibert_hopper_2d.sdf.

        :param hopper: A pydrake RigidBodyTree() loaded
            from raibert_hopper_2d.sdf.
        :param desired_lateral_velocity: How fast should the controller
            aim to run sideways?
        :param print_period: If nonzero, this controller will print to
            the python console every print_period (simulated) seconds
            to indicate simulation progress.
        '''
        VectorSystem.__init__(self,
            10, # 10 inputs: x, z, theta, alpha, l, and their derivatives
            2) # 2 outputs: Torque on thigh, and force on the leg extension
               #  link. (The passive spring for on the leg is calculated as
               #  part of this output.)
        self.hopper = hopper
        self.desired_lateral_velocity = desired_lateral_velocity
        self.print_period = print_period
        self.last_print_time = -print_period
        # Remember what the index of the foot is
        self.foot_body_index = hopper.FindBody("foot").get_body_index()

        # Default parameters for the hopper -- should match
        # raibert_hopper_1d.sdf, where applicable.
        # You're welcome to use these, but you probably don't need them.
        self.hopper_leg_length = 1.0
        self.m_b = 1.0
        self.m_f = 0.1
        self.l_max = 0.5

        # This is an arbitrary choice of spring constant for the leg.
        self.K_l = 100
           
    def ChooseSpringRestLength(self, X):
        '''
        Given the system state X,
            returns a (scalar) rest length of the leg spring.
            We can command this instantaneously, as
            the actual system being simulated has perfect
            force control of its leg extension.
        
        :param X: numpy array, length 10, full state of the hopper.

        :return: A float, the desired rest length of the hopper leg spring
                    to enforce.
        '''
        # Unpack states
        x, z, theta, alpha, l = X[0:5]
        zd = X[6]

        # Run out the forward kinematics of the robot
        # to figure out where the foot is in world frame.
        kinsol = self.hopper.doKinematics(X)
        foot_point = np.array([0.0, 0.0, -self.hopper_leg_length])
        foot_point_in_world = self.hopper.transformPoints(kinsol, 
                              foot_point, self.foot_body_index, 0)
        in_contact = foot_point_in_world[2] <= 0.00
        
        # Feel free to play with these values!
        # These should work pretty well for this problem set,
        # though.
        if (in_contact):
            if (zd > 0):
                # On the way back up,
                # "push" harder by increasing the effective
                # spring constant.
                l_rest = 1.15
            else:
                # On the way down,
                # "push" less hard by decreasing the effective
                # spring constant.
                l_rest = 1.0
        else:
            # Keep l_rest large to make sure the leg
            # is pushed back out to full extension quickly.
            l_rest = 1.0 

        # See "Hopping in Legged Systems-Modeling and
        # Simulation for the Two-Dimensional One-Legged Case"
        # Section III for a great description of why
        # this works. (It has to do with a natural balance
        # arising between the energy lost from dissipation
        # during ground contact, and the energy injected by
        # this control.)

        return l_rest

    def ChooseThighTorque(self, X):
        '''
        Given the system state X,
            returns a (scalar) rest length of the leg spring.
            We can command this instantaneously, as
            the actual system being simulated has perfect
            force control of its leg extension.
        
        :param X: numpy array, length 10, full state of the hopper.

        :return: A float, the torque to exert at the leg angle joint.
        '''
        x, z, theta, alpha, l = X[0:5]
        xd, zd, thetad, alphad, ld = X[5:10]
        
        # Run out the forward kinematics of the robot
        # to figure out where the foot is in world frame.
        kinsol = self.hopper.doKinematics(X)
        foot_point = np.array([0.0, 0.0, -self.hopper_leg_length])
        foot_point_in_world = self.hopper.transformPoints(kinsol, 
                              foot_point, self.foot_body_index, 0)
        in_contact = foot_point_in_world[2] <= 0.0
        
        K1 = .1
        K2 = .1
        K3 = .01
        xd_des = self.desired_lateral_velocity
        
        theta2 = theta
        theta2d = thetad

        theta2_des = 0
        
        if in_contact:
            return 1 * (theta2 - theta2_des) + .1 * theta2d

        r2 = 0
        
        r1 = self.hopper_leg_length
        w = r1
        M2 = self.m_b
        M1 = self.m_f
        
        xerr = K1 * (xd - xd_des) + K2 * (theta2 - theta2_des) + K3 * thetad
        
        #arg = (r2 * M2 * np.sin(theta2) + (M1+M2)*xerr)/(r1 * M1 + w * M2)
        arg = xerr
        
        theta1_desired = -np.arcsin(np.clip(arg, -0.8, 0.8))
        alpha_desired = theta1_desired - theta2
        #theta1 = alpha + theta2
        #theta1d = alphad + theta2d

        gain = 10
        return gain*(alpha_desired - alpha)
    
        # It's all yours from here.
        # Implement a controller that:
        #  - Controls xd to self.desired_lateral_velocity
        #  - Attempts to keep the body steady (theta = 0)
        return 0.0

    def _DoCalcVectorOutput(self, context, u, x, y):
        '''
        Given the state of the hopper (as the input to this system,
        u), populates (in-place) the control inputs to the hopper
        (y). This is given the state of this controller in x, but
        this controller has no state, so x is empty.

        :param u: numpy array, length 10, full state of the hopper.
        :param x: numpy array, length 0, full state of this controller.
        :output y: numpy array, length 2, control input to pass to the
            hopper.
        '''

        # The naming if inputs is confusing, as this is a separate
        # system with its own state (x) and input (u), but the input
        # here is the state of the hopper.

        l_rest = self.ChooseSpringRestLength(X = u)
        # Apply a force on the leg extension prismatic joint
        # that simulates the passive spring force (given the
        # chosen l_rest).
        leg_compression_amount = l_rest - u[4]
        
        # Print the current time, if requested,
        # as an indicator of how far simulation has
        # progressed.
        if (self.print_period and
            context.get_time() - self.last_print_time >= self.print_period):
            print "t: ", context.get_time()
            self.last_print_time = context.get_time()

        y[:] = [  self.ChooseThighTorque(X = u),
                  self.K_l * leg_compression_amount   ]


from pydrake.all import (DirectCollocation, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult)
from underactuated import (PlanarRigidBodyVisualizer)

from pydrake.all import (DiagramBuilder, FloatingBaseType, Simulator, VectorSystem,
                        ConstantVectorSource, SignalLogger, CompliantMaterial,
                         AddModelInstancesFromSdfString)
from IPython.display import HTML
import matplotlib.pyplot as plt

'''
Simulates a 2d hopper from initial conditions x0 (which
should be a 10x1 np array) for duration seconds,
targeting a specified lateral velocity and printing to the
console every print_period seconds (as an indicator of
progress, only if print_period is nonzero).
'''
def Simulate2dHopper(x0, duration,
        desired_lateral_velocity = 0.0,
        print_period = 0.0):
    builder = DiagramBuilder()

    # Load in the hopper from a description file.
    # It's spawned with a fixed floating base because
    # the robot description file includes the world as its
    # root link -- it does this so that I can create a robot
    # system with planar dynamics manually. (Drake doesn't have
    # a planar floating base type accessible right now that I
    # know about -- it only has 6DOF floating base types.)
    tree = RigidBodyTree()
    AddModelInstancesFromSdfString(
        open("raibert_hopper_2d.sdf", 'r').read(),
        FloatingBaseType.kFixed,
        None, tree)

    # A RigidBodyPlant wraps a RigidBodyTree to allow
    # forward dynamical simulation. It handles e.g. collision
    # modeling.
    plant = builder.AddSystem(RigidBodyPlant(tree))
    # Alter the ground material used in simulation to make
    # it dissipate more energy (to make the hopping more critical)
    # and stickier (to make the hopper less likely to slip).
    allmaterials = CompliantMaterial()
    allmaterials.set_youngs_modulus(1E8) # default 1E9
    allmaterials.set_dissipation(1.0) # default 0.32
    allmaterials.set_friction(1.0) # default 0.9.
    plant.set_default_compliant_material(allmaterials)

    # Spawn a controller and hook it up
    controller = builder.AddSystem(
        Hopper2dController(tree,
            desired_lateral_velocity = desired_lateral_velocity,
            print_period = print_period))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))

    # Create a logger to log at 30hz
    state_log = builder.AddSystem(SignalLogger(plant.get_num_states()))
    state_log._DeclarePeriodicPublish(0.0333, 0.0) # 30hz logging
    builder.Connect(plant.get_output_port(0), state_log.get_input_port(0))

    # Create a simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    # Don't limit realtime rate for this sim, since we
    # produce a video to render it after simulating the whole thing.
    #simulator.set_target_realtime_rate(100.0) 
    simulator.set_publish_every_time_step(False)

    # Force the simulator to use a fixed-step integrator,
    # which is much faster for this stiff system. (Due to the
    # spring-model of collision, the default variable-timestep
    # integrator will take very short steps. I've chosen the step
    # size here to be fast while still being stable in most situations.)
    integrator = simulator.get_mutable_integrator()
    integrator.set_fixed_step_mode(True)
    integrator.set_maximum_step_size(0.0005)

    # Set the initial state
    state = simulator.get_mutable_context().get_mutable_continuous_state_vector()
    state.SetFromVector(x0)

    # Simulate!
    simulator.StepTo(duration)

    return tree, controller, state_log
