import hybrid_automaton as ha

def vel_des_rect2pol(vel_des_rect, omega_max, phi_guess):
    """convert desired velocity from rectangular to polar coords"""
    vx_des = vel_des_rect[0]
    vy_des = vel_des_rect[1]
    phi_des = np.arctan2(vy_des, vx_des)
    phi_guess = pos_guess[2]
    vel_des_phi = omega_max * (phi_des % (2*pi) - phi_guess % (2*pi))
    vel_des_pol = (vel_des_r, vel_des_phi)
    return vel_des_pol

## Policies

def gtg_policy(slowdown_radius, vel_max, vel_guess, displacement):
    """control policy for go to goal behavior"""
    displacement_norm = np.linalg.norm(displacement)
    slowdown_factor = (1 - exp(-displacement_norm / slowdown_radius))
    vel_des_r = vel_max * slowdown_factor
    vel_des_rect = vel_des_r * displacement / displacement_norm
    vel_des_pol = vel_des_rect2pol(vel_des_rect)
    control_v = np.reshape(vel_des_pol - vel_guess, (2, 1))
    return control_v

def ao_policy(vel_guess, flee_vector):
    """control_policy for avoid obstacle behavior"""
    control_x = np.array([[0],[0],[0]])
    vel_des_pol = vel_des_rect2pol(vel_max * flee_vector)
    control_v = np.reshape(vel_des_pol - vel_guess, (2, 1))
    return control_v

def fw_cc_policy(vel_guess, flee_vector):
    """control policy for follow wall counter clockwise behavior"""
    flee_x, flee_y = flee_vector
    fw_cc_vector = (flee_y, -flee_x)
    vel_des_pol = vel_des_rect2pol(vel_max * fw_cc_vector)
    control_v = np.reshape(vel_des_pol - vel_guess, (2, 1))
    return control_v

def fw_c_policy(vel_guess, flee_vector):
    """control policy for follow wall clockwise behavior"""
    flee_x, flee_y = flee_vector
    fw_c_vector = (-flee_y, flee_x)
    vel_des_pol = vel_des_rect2pol(vel_max * fw_c_vector)
    control_v = np.reshape(vel_des_pol - vel_guess, (2, 1))
    return control_v

def goal_policy():
    """control policy for goal reached behavior"""
    vel_des_pol = (0,0)
    control_v = np.reshape(vel_des_pol - vel_guess, (2, 1))
    return control_v

## Guard Conditions

def condition_fw_gtg(goal_vector, flee_vector, last_goal_distance):
    """condition for transition from follow wall to go-to-goal"""
    distance = np.linalg.norm(goal_vector)
    direction = np.dot(goal_vector, flee_vector)
    return (distance < last_goal_distance and direction> 0)

def condition_fw_ao(, flee_threshold, guard_fatness):
    """condition for transision from follow wall to avoid obstacle"""
    return (obst_distance < flee_threshold - 0.5*guard_fatness)

def condition_gtg_fw_cc(goal_vector, flee_vector, flee_threshold, guard_fatness):
    """condition for transition from go-to-goal to follow wall
    counter-clockwise"""
    flee_x, flee_y = flee_vector
    fw_cc_vector = (flee_y, -flee_x)
    direction = np.dot(goal_vector, fw_cc_vector)
    return (obst_distance < flee_threshold + 0.5*guard_fatness and direction >
            0)

def condition_gtg_fw_c(goal_vector, flee_vector, flee_threshold, guard_fatness):
    """condition for transition from go-to-goal to follow wall
    clockwise"""
    flee_x, flee_y = flee_vector
    fw_c_vector = (-flee_y, flee_x)
    direction = np.dot(goal_vector, fw_c_vector)
    return (obst_distance < flee_threshold + 0.5*guard_fatness and direction >
            0)

def condition_ao_fw_cc(goal_vector, flee_vector, flee_threshold, guard_fatness):
    """condition for transition from avoid obstacle to follow wall
    counter-clockwise"""
    flee_x, flee_y = flee_vector
    fw_cc_vector = (flee_y, -flee_x)
    direction = np.dot(goal_vector, fw_cc_vector)
    return (obst_distance < flee_threshold - 0.5*guard_fatness and direction >
            0)


def condition_ao_fw_c(goal_vector, flee_vector, flee_threshold, guard_fatness):
    """condition for transition from avoid obstacle to follow wall
    clockwise"""
    flee_x, flee_y = flee_vector
    fw_c_vector = (-flee_y, flee_x)
    direction = np.dot(goal_vector, fw_c_vector)
    return (obst_distance < flee_threshold - 0.5*guard_fatness and direction >
            0)

def condition_gtg_goal(goal_vector, goal_radius):
    """condition for transition from go-to-goal to goal"""
    distance = np.linalg.norm(goal_vector)
    return (distance < goal_radius)

## Resets

def record_goal_distance(state, goal_distance):
    """record the goal distance upon entering a follow wall behavior"""
    state['lgd'] = goal_distance

def no_reset(state):
    pass

## Behaviors

behavior_gtg = ha.Behavior(gtg_policy)
behavior_ao = ha.Behavior(ao_policy)
behavior_fw_cc = ha.Behavior(fw_cc_policy)
behavior_fw_c = ha.Behavior(fw_c_policy)
behavior_goal = ha.Behavior(goal_policy)

## Guards

guard_gtg_fw_cc = Guard(condition_gtg_fw_cc, behavior_fw_cc,
        record_goal_distance)
guard_gtg_fw_c  = Guard(condition_gtg_fw_c, behavior_fw_c, record_goal_distance)
guard_gtg_goal  = Guard(condition_gtg_goal, behavior_goal, no_reset)
behavior_gtg.add_guards([guard_gtg_fw_cc, guard_gtg_fw_c, guard_gtg_goal])

guard_fw_cc_gtg = Guard(condition_fw_gtg, behavior_gtg, no_reset)
guard_fw_cc_ao  = Guard(condition_fw_ao, behavior_ao, no_reset)
behavior_fw_cc.add_guards([guard_fw_cc_gtg, guard_fw_cc_ao])

guard_fw_c_gtg  = Guard(condition_fw_gtg, behavior_gtg,no_reset)
guard_fw_c_ao   = Guard(condition_fw_ao, behavior_ao, no_reset)
behavior_fw_c.add_guards([guard_fw_c_gtg, guard_fw_c_ao])

guard_ao_fw_cc  = Guard(condition_ao_fw_cc, behavior_fw_cc,
        record_goal_distance)
guard_ao_fw_c   = Guard(condition_ao_fw_c, behavior_fw_c, record_goal_distance)
behavior_ao.add_guards([guard_ao_fw_cc, guard_ao_fw_c])

this_ha = ha.HybridAutomaton([behavior_gtg, behavior_ao, behavior_fw_cc,
    behavior_fw_c], behavior_gtg, {})
