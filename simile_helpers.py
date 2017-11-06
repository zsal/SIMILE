
def joint_loss_calculation(human_trajectory, predicted_trajectory, penalize):
    acceleration = np.zeros(human_trajectory.shape)
    for item in inPlay:
        acceleration[item[0]+2:item[1]+1] = np.diff(np.diff(predicted_trajectory[item[0]:item[1]+1]))
    #jointLoss = math.sqrt(((new_learned_trajectory-human)**2).mean())
    absLoss = np.mean(((predicted_trajectory-human_trajectory)**2))
    smoothLoss = penalize*np.mean(acceleration**2)
    jointLoss = math.sqrt(absLoss + smoothLoss)
    return jointLoss

def test_error_calculation(human_trajectory, predicted_trajectory, penalize):
    acceleration = np.zeros(human_trajectory.shape[0])
    acceleration[2:] = np.diff(np.diff(predicted_trajectory))
    #jointLoss = math.sqrt(((new_learned_trajectory-human)**2).mean())
    absLoss = np.mean(((predicted_trajectory-human_trajectory)**2))
    smoothLoss = penalize*np.mean(acceleration**2)
    jointLoss = absLoss + smoothLoss
    return jointLoss    

def interpolate_learned_policy(old_policy, new_policy, interpolate, old_coeff, new_coeff, weight, method):
    if method is "stack_vel_pos":
        learned_trajectory = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                learned_trajectory[index] = human[index]
            for index in np.arange(item[0]+tao,item[1]+1):
                feature = autoreg_game_context[index,:]
                for i in range(tao-1):
                    feature = np.append(feature, learned_trajectory[index-(i+1)] - learned_trajectory[index-(i+2)])
                for i in range(tao):
                    feature = np.append(feature,learned_trajectory[index-(i+1)])
                previous_prediction = learned_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                old_model_predict = (old_policy.predict(feature) + np.inner(old_coeff, previous_prediction) * weight) / (1+weight)
                new_model_predict = (new_policy.predict(feature) + np.inner(new_coeff, previous_prediction) * weight) / (1+weight)
                #current_prediction = interpolate * new_policy.predict(feature) + (1-interpolate) * old_policy.predict(feature)
                learned_trajectory[index] = interpolate * new_model_predict + (1-interpolate) * old_model_predict
    return learned_trajectory    

def interpolate_test_policy(old_policy, new_policy, interpolate, reference_path, context, old_coeff, new_coeff, weight, method):
    Y_predict = np.zeros(reference_path.shape)
    if method is "stack_vel_pos":
        for i in range(len(reference_path)):
            if i<tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
            else:
                feature = context[i]
                for j in range(tao-1):
                    feature = np.hstack((feature,Y_predict[i-(j+1)]-Y_predict[i-(j+2)]))
                for j in range(tao):
                    feature = np.hstack((feature,Y_predict[i-(j+1)]))
                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                #current_prediction = interpolate * new_policy.predict(feature) + (1-interpolate) * old_policy.predict(feature)
                old_model_predict = (old_policy.predict(feature) + np.inner(old_coeff, previous_prediction) * weight) / (1+weight)
                new_model_predict = (new_policy.predict(feature) + np.inner(new_coeff, previous_prediction) * weight) / (1+weight)
                #Y_predict[i] = (current_prediction + np.inner(coeff,previous_prediction)*weight)/(1+weight) # replace
                Y_predict[i] = interpolate * new_model_predict + (1-interpolate) * old_model_predict
    return Y_predict    

def collect_learned_trajectory(policy, coeff, weight, roll_method):
    if roll_method is "stack_pos":
        learned_trajectory = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                learned_trajectory[index] = human[index]
            for index in np.arange(item[0]+tao,item[1]+1):
                feature = autoreg_game_context[index,:]
                for i in range(tao):
                    feature = np.append(feature,learned_trajectory[index-(i+1)])
                previous_prediction = learned_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                learned_trajectory[index] = (policy.predict(feature)+ np.inner(coeff,
                                             previous_prediction)*weight) / (1+weight) # replace
    elif roll_method is "stack_pos_vel":
        learned_trajectory = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                learned_trajectory[index] = human[index]
            for index in np.arange(item[0]+tao,item[1]+1):
                feature = autoreg_game_context[index,:]
                for i in range(tao):
                    feature = np.append(feature,learned_trajectory[index-(i+1)])
                for i in range(tao-1):
                    feature = np.append(feature, learned_trajectory[index-(i+1)] - learned_trajectory[index-(i+2)])
                previous_prediction = learned_trajectory[index-tao:index].copy()
                previous_velocity = np.diff(previous_prediction)
                previous_prediction = previous_prediction[::-1]
                previous_velocity = previous_velocity[::-1]
                dependent_feature = np.concatenate((previous_prediction, previous_velocity))
                learned_trajectory[index] = (policy.predict(feature)+ np.inner(coeff,
                                             dependent_feature)*weight) / (1+weight) # replace
    elif roll_method is "stack_vel_pos":
        learned_trajectory = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                learned_trajectory[index] = human[index]
            for index in np.arange(item[0]+tao,item[1]+1):
                feature = autoreg_game_context[index,:]
                for i in range(tao-1):
                    feature = np.append(feature, learned_trajectory[index-(i+1)] - learned_trajectory[index-(i+2)])
                for i in range(tao):
                    feature = np.append(feature,learned_trajectory[index-(i+1)])
                previous_prediction = learned_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                learned_trajectory[index] = (policy.predict(feature)+ np.inner(coeff,
                                             previous_prediction)*weight) / (1+weight) # replace
    return learned_trajectory

def collect_learned_trajectory_position_velocity(policy, coeff, weight):
    learned_trajectory = np.zeros(human.shape)
    for item in inPlay:
        for index in np.arange(item[0],item[0]+tao):
            learned_trajectory[index] = human[index]
        for index in np.arange(item[0]+tao,item[1]+1):
            feature = autoreg_game_context[index,:]
            for i in range(tao):
                feature = np.append(feature,learned_trajectory[index-(i+1)])
            for i in range(tao-1):
                feature = np.append(feature, learned_trajectory[index-(i+1)] - learned_trajectory[index-(i+2)])
            previous_prediction = learned_trajectory[index-tao:index].copy()
            previous_velocity = np.diff(previous_prediction)
            previous_prediction = previous_prediction[::-1]
            previous_velocity = previous_velocity[::-1]
            dependent_feature = np.concatenate((previous_prediction, previous_velocity))
            learned_trajectory[index] = (policy.predict(feature)+ np.inner(coeff,
                                         dependent_feature)*weight) / (1+weight) # replace
    return learned_trajectory

def collect_test_trajectory(policy, reference_path, context, coeff, weight, method):
    Y_predict = np.zeros(reference_path.shape)
    if method is "stack_pos":
        for i in range(len(reference_path)):
            if i<tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
            else:
                feature = context[i]
                for j in range(tao):
                    feature = np.hstack((feature,Y_predict[i-(j+1)]))
                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                Y_predict[i] = (policy.predict(feature)[0] + np.inner(coeff,previous_prediction)*weight)/(1+weight) # replace
    elif method is "stack_pos_vel":
        for i in range(len(reference_path)):
            if i<tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
            else:
                feature = context[i]
                for j in range(tao):
                    feature = np.hstack((feature,Y_predict[i-(j+1)]))
                for j in range(tao-1):
                    feature = np.hstack((feature,Y_predict[i-(j+1)]-Y_predict[i-(j+2)]))
                previous_prediction = Y_predict[i-tao:i]
                previous_velocity = np.diff(previous_prediction)
                previous_prediction = previous_prediction[::-1]
                previous_velocity = previous_velocity[::-1]
                dependent_feature = np.concatenate((previous_prediction, previous_velocity))
                Y_predict[i] = (policy.predict(feature)[0] + np.inner(coeff,dependent_feature)*weight)/(1+weight) # replace
    elif method is "stack_vel_pos":
        for i in range(len(reference_path)):
            if i<tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
            else:
                feature = context[i]
                for j in range(tao-1):
                    feature = np.hstack((feature,Y_predict[i-(j+1)]-Y_predict[i-(j+2)]))
                for j in range(tao):
                    feature = np.hstack((feature,Y_predict[i-(j+1)]))
                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                Y_predict[i] = (policy.predict(feature)[0] + np.inner(coeff,previous_prediction)*weight)/(1+weight) # replace
    return Y_predict

def collect_test_trajectory_position_velocity(policy, reference_path, context, coeff, weight):
    Y_predict = np.zeros(reference_path.shape)
    for i in range(len(reference_path)):
        if i<tao:
            Y_predict[i] = reference_path[i] #note: have the first tau frames correct
        else:
            feature = context[i]
            for j in range(tao):
                feature = np.hstack((feature,Y_predict[i-(j+1)]))
            for j in range(tao-1):
                feature = np.hstack((feature,Y_predict[i-(j+1)]-Y_predict[i-(j+2)]))
            previous_prediction = Y_predict[i-tao:i]
            previous_velocity = np.diff(previous_prediction)
            previous_prediction = previous_prediction[::-1]
            previous_velocity = previous_velocity[::-1]
            dependent_feature = np.concatenate((previous_prediction, previous_velocity))
            Y_predict[i] = (policy.predict(feature)[0] + np.inner(coeff,dependent_feature)*weight)/(1+weight) # replace
    return Y_predict

def calculate_smooth_coeff(trajectory, reg_alpha, back_horizon, method):
    if method is "stack_pos":
        smooth_coeff = velocity_smooth(trajectory, reg_alpha, back_horizon)
    elif method is "stack_pos_proper":
        smooth_coeff = velocity_smooth(trajectory, reg_alpha, back_horizon)
    elif method is "stack_vel_pos":
        smooth_coeff = velocity_smooth(trajectory, reg_alpha, back_horizon)
    elif method is "stack_pos_vel":
        smooth_coeff = position_and_velocity_smooth(trajectory, reg_alpha, back_horizon)
    elif method is "pos_based":
        smooth_coeff = position_smooth(trajectory, reg_alpha, back_horizon)
    elif method is "stack_pos_res":
        smooth_coeff = velocity_smooth(trajectory, reg_alpha, back_horizon)
    elif method is "stack_pos_res_proper" or "stack_pos_res_predict":
        smooth_coeff = velocity_smooth(trajectory, reg_alpha, back_horizon)
    elif method is "no_stack":
        smooth_coeff = velocity_smooth(trajectory, reg_alpha, back_horizon)
    elif method is "predict_pos_res":
        smooth_coeff = velocity_smooth(trajectory, reg_alpha, back_horizon)
    return smooth_coeff

def velocity_smooth(trajectory, reg_alpha, back_horizon):

    clf = linear_model.Ridge(alpha = reg_alpha)

    velocity_ar_seg = np.empty(shape = [trajectory.shape[0],back_horizon-1]) #initialize an empty array to hold the autoregressed velocity values

    velocity = np.empty(shape = trajectory.shape)
    for item in inPlay:
        #v_ar = np.empty([item[1]-item[0]+1,0])
        #velocity[item[0]:(item[1]+1)] = np.gradient(Y[item[0]:(item[1]+1)])
        velocity[item[0]:(item[1]+1)] = np.insert(np.diff(trajectory[item[0]:(item[1]+1)]),
                                                  0,trajectory[item[0]]) # calculate backward difference
        for i in range(back_horizon-1):
            temp = np.roll(velocity[item[0]:(item[1]+1)],i+1)
            for j in range(i+1):
                temp[j] = 0
            velocity_ar_seg[item[0]:(item[1]+1),i] = temp.copy()
    rows_to_delete = []
    for item in inPlay:
        for i in range(back_horizon-1): # change back_horizon-1 to back_horizon
            rows_to_delete.append(item[0]+i)
        #for i in range(back_horizon):
        #    velocity = np.delete(velocity,item[0],0)
        #    velocity_ar_seg = np.delete(velocity_ar_seg,item[0],0)
    velocity = np.delete(velocity,rows_to_delete,0)
    velocity_ar_seg = np.delete(velocity_ar_seg, rows_to_delete,0)
    # Use least square regression to find the best fit set of coefficients for the velocity vectors
    #velocity_smooth_interpolate = np.linalg.lstsq(velocity_ar_seg,velocity)[0] 
    clf.fit(velocity_ar_seg, velocity)
    velocity_smooth_interpolate = clf.coef_
    # Build position smooth coefficients out of velocity smooth coeeficient array
    smooth_coeff = np.array([1+velocity_smooth_interpolate[0]])
    for i in range(back_horizon-2):
        smooth_coeff = np.append(smooth_coeff,velocity_smooth_interpolate[i+1]-velocity_smooth_interpolate[i])
    smooth_coeff = np.append(smooth_coeff,-velocity_smooth_interpolate[back_horizon-2])

    return smooth_coeff

def position_and_velocity_smooth(trajectory, reg_alpha, back_horizon):
    position_ar_seg = np.empty(shape = [trajectory.shape[0],back_horizon]) #initialize an empty array to hold the autoregressed position values
    position = trajectory.copy() #initialize position vector to simply be the output vector
    for item in inPlay:
        for i in range(back_horizon):
            temp = np.roll(position[item[0]:(item[1]+1)],i+1)
            for j in range(i+1):
                temp[j] = 0
            position_ar_seg[item[0]:(item[1]+1),i] = temp.copy()

    velocity_ar_seg = np.empty(shape = [trajectory.shape[0],back_horizon-1]) #initialize an empty array to hold the autoregressed velocity values
    velocity = np.empty(shape = trajectory.shape)

    for item in inPlay:
        velocity[item[0]:(item[1]+1)] = np.insert(np.diff(position[item[0]:(item[1]+1)]),
                                                  0,position[item[0]]) # calculate backward difference
        for i in range(back_horizon-1):
            temp = np.roll(velocity[item[0]:(item[1]+1)],i+1)
            for j in range(i+1):
                temp[j] = 0
            velocity_ar_seg[item[0]:(item[1]+1),i] = temp.copy()
    rows_to_delete = []
    for item in inPlay:
        for i in range(tao):
            rows_to_delete.append(item[0]+i)
    position = np.delete(position, rows_to_delete,0)
    position_ar_seg = np.delete(position_ar_seg, rows_to_delete,0)
    velocity_ar_seg = np.delete(velocity_ar_seg, rows_to_delete,0)

    X = np.hstack((position_ar_seg, velocity_ar_seg))
    clf = linear_model.Ridge(alpha = reg_alpha)
    clf.fit(X, position)
    smooth_coeff = clf.coef_
    return smooth_coeff

def position_smooth(trajectory, reg_alpha, back_horizon):
    # Alternative method to calculate the smooth coefficients: try to fit y-values directly to explain smoothness
    clf = linear_model.Ridge(alpha = reg_alpha)
    lookahead = 1 # will change this globally if we look into looking ahead further in the future
    position_ar_seg = np.empty(shape = [trajectory.shape[0],back_horizon]) #initialize an empty array to hold the autoregressed position values
    position = trajectory.copy() #initialize position vector to simply be the output vector
    for item in inPlay:
        for i in range(back_horizon):
            temp = np.roll(position[item[0]:(item[1]+1)],i+1)
            for j in range(i+1):
                temp[j] = 0
            position_ar_seg[item[0]:(item[1]+1),i] = temp.copy()
        for i in range(lookahead-1):
            temp = np.roll(position[item[0]:(item[1]+1)],-(i+1)) 
            for j in range(i+1):
                temp[-(j+1)] = 0
            position_ar_seg[item[0]:(item[1]+1),i+tao] = temp.copy()
    rows_to_delete = []
    for item in inPlay:
        for i in range(tao):
            rows_to_delete.append(item[0]+i)
        for i in range(lookahead-1):
            rows_to_delete.append(item[1]-i)
    position = np.delete(position, rows_to_delete,0)
    position_ar_seg = np.delete(position_ar_seg, rows_to_delete,0)
    # Use least square regression to find the best fit set of coefficients for the velocity vectors
    #position_smooth_interpolate = np.linalg.lstsq(position_ar_seg,position)[0]
    #Note that in practice, the outcome of position_smooth_coeff and position_smooth_interpolate seem to be quite similar
    clf.fit(position_ar_seg,position) # addition to switch from velocity to position
    position_smooth_interpolate = clf.coef_ # addition to switch from velocity to position
    return position_smooth_interpolate

def residual_smooth(trajectory, reg_alpha, back_horizon):
    # Alternative method to calculate the smooth coefficients: try to fit y-values directly to explain smoothness
    clf = linear_model.Ridge(alpha = reg_alpha)
    residual_ar_seg = np.empty(shape = [trajectory.shape[0],back_horizon]) #initialize an empty array to hold the autoregressed position values
    residual = trajectory.copy() #initialize position vector to simply be the output vector
    for item in inPlay:
        for i in range(back_horizon):
            temp = np.roll(residual[item[0]:(item[1]+1)],i+1)
            for j in range(i+1):
                temp[j] = 0
            residual_ar_seg[item[0]:(item[1]+1),i] = temp.copy()
    rows_to_delete = []
    for item in inPlay:
        for i in range(2*back_horizon):
            rows_to_delete.append(item[0]+i)
    residual = np.delete(residual, rows_to_delete,0)
    residual_ar_seg = np.delete(residual_ar_seg, rows_to_delete,0)
    # Use least square regression to find the best fit set of coefficients for the velocity vectors
    #position_smooth_interpolate = np.linalg.lstsq(position_ar_seg,position)[0]
    #Note that in practice, the outcome of position_smooth_coeff and position_smooth_interpolate seem to be quite similar
    clf.fit(residual_ar_seg,residual) # addition to switch from velocity to position
    residual_smooth_interpolate = clf.coef_ # addition to switch from velocity to position
    return residual_smooth_interpolate

def residual_diff_smooth(trajectory, reg_alpha, back_horizon):
    clf = linear_model.Ridge(alpha = reg_alpha)

    velocity_ar_seg = np.empty(shape = [trajectory.shape[0],back_horizon-1]) #initialize an empty array to hold the autoregressed velocity values

    velocity = np.empty(shape = trajectory.shape)
    for item in inPlay:
        #v_ar = np.empty([item[1]-item[0]+1,0])
        #velocity[item[0]:(item[1]+1)] = np.gradient(Y[item[0]:(item[1]+1)])
        velocity[item[0]:(item[1]+1)] = np.insert(np.diff(trajectory[item[0]:(item[1]+1)]),
                                                  0,trajectory[item[0]]) # calculate backward difference
        for i in range(back_horizon-1):
            temp = np.roll(velocity[item[0]:(item[1]+1)],i+1)
            for j in range(i+1):
                temp[j] = 0
            velocity_ar_seg[item[0]:(item[1]+1),i] = temp.copy()
    rows_to_delete = []
    for item in inPlay:
        for i in range(2*back_horizon): # change back_horizon-1 to 2*back_horizon
            rows_to_delete.append(item[0]+i)
        #for i in range(back_horizon):
        #    velocity = np.delete(velocity,item[0],0)
        #    velocity_ar_seg = np.delete(velocity_ar_seg,item[0],0)
    velocity = np.delete(velocity,rows_to_delete,0)
    velocity_ar_seg = np.delete(velocity_ar_seg, rows_to_delete,0)
    # Use least square regression to find the best fit set of coefficients for the velocity vectors
    #velocity_smooth_interpolate = np.linalg.lstsq(velocity_ar_seg,velocity)[0] 
    clf.fit(velocity_ar_seg, velocity)
    velocity_smooth_interpolate = clf.coef_
    # Build position smooth coefficients out of velocity smooth coeeficient array
    smooth_coeff = np.array([1+velocity_smooth_interpolate[0]])
    for i in range(back_horizon-2):
        smooth_coeff = np.append(smooth_coeff,velocity_smooth_interpolate[i+1]-velocity_smooth_interpolate[i])
    smooth_coeff = np.append(smooth_coeff,-velocity_smooth_interpolate[back_horizon-2])

    return smooth_coeff    

def form_augmented_context_train(X,back_horizon):
    augmented_X = np.empty([0,(back_horizon+1)*number_of_feature]) 
    for item in inPlay:
        segment = X[item[0]:(item[1]+1),:]
        autoreg_segment = segment.copy()
        for i in range(back_horizon):
            temp = np.roll(segment,(i+1)*number_of_feature)
            for j in range(i+1):
                temp[j] = 0
            autoreg_segment = np.hstack((autoreg_segment,temp))
        augmented_X = np.vstack((augmented_X,autoreg_segment))
    return augmented_X

def form_auxiliary_position_vector(trajectory, back_horizon):
    auxiliary_previous_output = np.empty([0,back_horizon]) #Auxiliary signals

    for item in inPlay:
        explore_segment = trajectory[item[0]:(item[1]+1)]
        auxiliary_segment = np.empty([item[1]-item[0]+1,0])
        for i in range(back_horizon):
            temp = np.roll(explore_segment,i+1)
            for j in range(i+1):
                temp[j] = 0
            auxiliary_segment = np.hstack((auxiliary_segment,np.vstack(temp)))
        auxiliary_previous_output = np.vstack((auxiliary_previous_output,auxiliary_segment))
    return auxiliary_previous_output

def form_auxiliary_velocity_vector(trajectory, back_horizon):
    velocity_ar_seg = np.empty(shape = [trajectory.shape[0],back_horizon-1]) #initialize an empty array to hold the autoregressed velocity values
    velocity = np.empty(shape = trajectory.shape)

    for item in inPlay:
        velocity[item[0]:(item[1]+1)] = np.insert(np.diff(trajectory[item[0]:(item[1]+1)]),
                                                  0,trajectory[item[0]]) # calculate backward difference
        for i in range(back_horizon-1):
            temp = np.roll(velocity[item[0]:(item[1]+1)],i+1)
            for j in range(i+1):
                temp[j] = 0
            velocity_ar_seg[item[0]:(item[1]+1),i] = temp.copy()
    return velocity_ar_seg

def gather_rows_to_delete(back_horizon):
    rows_to_delete = []
    if smooth_method is "stack_pos_res":
        for item in inPlay:
            for i in range(2*tao):
                rows_to_delete.append(item[0]+i)
    elif smooth_method is "stack_pos_res_proper" or "stack_pos_res_predict":
        for item in inPlay:
            for i in range(2*tao):
                rows_to_delete.append(item[0]+i)
    elif smooth_method is "predict_pos_res":
        for item in inPlay:
            for i in range(tao+tao):
                rows_to_delete.append(item[0]+i)    
    else:
        for item in inPlay:
            for i in range(tao-1):
                rows_to_delete.append(item[0]+i)
    return rows_to_delete

def equivalent_position_coeff(pos_vel_smooth_coeff, back_horizon):
    smooth_coeff = np.array([pos_vel_smooth_coeff[0]+pos_vel_smooth_coeff[back_horizon]])
    for i in range(back_horizon-2):
        smooth_coeff = np.append(smooth_coeff, pos_vel_smooth_coeff[i+1] - pos_vel_smooth_coeff[i+back_horizon]+pos_vel_smooth_coeff[i+back_horizon+1])
    smooth_coeff = np.append(smooth_coeff, pos_vel_smooth_coeff[back_horizon-1] -pos_vel_smooth_coeff[2*back_horizon-2])
    return smooth_coeff

def form_state_vector(augmented_X, trajectory, back_horizon, method):
    if (method is "stack_pos") or (method is "stack_pos_proper"):
        state_vectors = form_state_vector_pos(augmented_X, trajectory, back_horizon)
    elif method is "stack_pos_vel":
        aux_position = form_auxiliary_position_vector(trajectory, back_horizon)
        aux_velocity = form_auxiliary_velocity_vector(trajectory, back_horizon)
        state_vectors = np.hstack((augmented_X,aux_position,aux_velocity))
    elif method is "stack_vel_pos":
        aux_position = form_auxiliary_position_vector(trajectory, back_horizon)
        aux_velocity = form_auxiliary_velocity_vector(trajectory, back_horizon)
        state_vectors = np.hstack((augmented_X,aux_velocity,aux_position))
    return state_vectors

def form_state_vector_pos(augmented_X, trajectory, back_horizon):
    state_vectors_explore = augmented_X.copy() #Initialize states, unaugmented with Y's initially, before attaching Y's on
    auxiliary_previous_output = np.empty([0,back_horizon]) #Auxiliary signals

    for item in inPlay:
        explore_segment = trajectory[item[0]:(item[1]+1)]
        auxiliary_segment = np.empty([item[1]-item[0]+1,0])
        for i in range(back_horizon):
            temp = np.roll(explore_segment,i+1)
            for j in range(i+1):
                temp[j] = 0
            auxiliary_segment = np.hstack((auxiliary_segment,np.vstack(temp)))
        auxiliary_previous_output = np.vstack((auxiliary_previous_output,auxiliary_segment))
    state_vectors_explore = np.hstack((state_vectors_explore, auxiliary_previous_output))
    return state_vectors_explore

def calculate_residual(trajectory, back_horizon, coeff):
    residual = np.zeros(trajectory.shape)
    position_ar_seg = np.empty(shape = [trajectory.shape[0],back_horizon])
    position = trajectory.copy()
    for item in inPlay:
        for i in range(back_horizon):
            temp = np.roll(position[item[0]:(item[1]+1)],i+1)
            position_ar_seg[item[0]:(item[1]+1),i] = temp.copy()
    for item in inPlay:
        residual[item[0]:(item[1]+1)] = trajectory[item[0]:(item[1]+1)] - np.dot(position_ar_seg[item[0]:(item[1]+1),:], coeff)
    #residual = trajectory - np.dot(position_ar_seg, coeff)
    for item in inPlay:
       for i in range(back_horizon):
           residual[item[0]+i] = 0
    return residual

def rollout_nofilter_learned_trajectory(policy, coeff, method):
    if (method is "stack_pos") or (method is "stack_pos_proper"):
        learned_trajectory = np.zeros(human.shape)

        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                learned_trajectory[index] = human[index]
            for index in np.arange(item[0]+tao,item[1]+1):
                feature = autoreg_game_context[index,:]
                for i in range(tao):
                    feature = np.append(feature,learned_trajectory[index-(i+1)])
                learned_trajectory[index] = policy.predict(feature)
    elif method is "stack_vel_pos":
        learned_trajectory = np.zeros(human.shape)

        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                learned_trajectory[index] = human[index]
            for index in np.arange(item[0]+tao,item[1]+1):
                feature = autoreg_game_context[index,:]
                for i in range(tao-1):
                    feature = np.append(feature, learned_trajectory[index-(i+1)] - learned_trajectory[index-(i+2)])
                for i in range(tao):
                    feature = np.append(feature,learned_trajectory[index-(i+1)])
                #previous_prediction = learned_trajectory[index-tao:index].copy()
                #previous_prediction = previous_prediction[::-1]
                learned_trajectory[index] = policy.predict(feature)
                #learned_trajectory[index] = (policy.predict(feature)+ np.inner(coeff,
                #                             previous_prediction)*weight) / (1+weight) # replace
    elif method is ("stack_pos_res" or "stack_pos_res_proper" or "stack_pos_res_predict"):
        learned_trajectory = np.zeros(human.shape)

        for item in inPlay:
            for index in np.arange(item[0],item[0]+2*tao):
                learned_trajectory[index] = human[index]
            for index in np.arange(item[0]+2*tao, item[1]+1):
                feature = autoreg_game_context[index,:]
                # stack position features
                for i in range(tao):
                    feature = np.append(feature,learned_trajectory[index-(i+1)])
                # stack residual features
                for i in range(tao):
                    previous_position = learned_trajectory[index-(i+1)-tao:index-(i+1)]
                    previous_position = previous_position[::-1]
                    feature = np.append(feature, learned_trajectory[index-(i+1)] - np.inner(coeff[:tao],previous_position)) #??? shouldn't be coeff[tao:]
                learned_trajectory[index] = policy.predict(feature)

    return learned_trajectory

def rollout_nofilter_test_trajectory(policy, reference_path, context, coeff, method):
    if (method is "stack_pos") or (method is "stack_pos_proper"):
        Y_roll = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<tao:
                Y_roll[i] = reference_path[i] #note: have the first tau frames correct
            else:
                feature = context[i]
                for j in range(tao):
                    feature = np.hstack((feature,Y_roll[i-(j+1)]))
                Y_roll[i] = policy.predict(feature)[0]
    elif method is "stack_vel_pos":
        Y_roll = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<tao:
                Y_roll[i] = reference_path[i] #note: have the first tau frames correct
            else:
                feature = context[i]
                for j in range(tao-1):
                    feature = np.hstack((feature,Y_roll[i-(j+1)]-Y_roll[i-(j+2)]))
                for j in range(tao):
                    feature = np.hstack((feature,Y_roll[i-(j+1)]))
                #previous_prediction = Y_predict[i-tao:i]
                #previous_prediction = previous_prediction[::-1]
                Y_roll[i] = policy.predict(feature)[0]
                #Y_predict[i] = (policy.predict(feature)[0] + np.inner(coeff,previous_prediction)*weight)/(1+weight) # replace
    elif method is ("stack_pos_res" or "stack_pos_res_proper" or "stack_pos_res_predict"):   
        Y_roll = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<2*tao:
                Y_roll[i] = reference_path[i]
            else:
                feature = context[i]
                #stack position features
                for j in range(tao):
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]))
                    feature = np.append(feature, Y_roll[i-(j+1)])
                #stack residual features
                for j in range(tao):
                    previous_position = Y_roll[i-(j+1)-tao:i-(j+1)]
                    previous_position = previous_position[::-1]
                    feature = np.append(feature, Y_roll[i-(j+1)] - np.inner(coeff[0:tao], previous_position))
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]-Y_roll[i-(j+2)]))
                Y_roll[i] = policy.predict(feature)[0]
    return Y_roll

def roll_and_smooth_learned_trajectory(policy, coeff, weight, method):
    if method is "stack_vel_pos":
        roll_trajectory = np.zeros(human.shape)
        filtered_trajectory = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                roll_trajectory[index] = human[index]
                filtered_trajectory[index] = human[index]
            for index in np.arange(item[0]+tao, item[1]+1):
                feature = autoreg_game_context[index,:]
                for i in range(tao-1):
                    feature = np.append(feature, roll_trajectory[index-(i+1)] - roll_trajectory[index-(i+2)])
                for i in range(tao):
                    feature = np.append(feature,roll_trajectory[index-(i+1)])
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                roll_trajectory[index] = policy.predict(feature)
                filtered_trajectory[index] = (roll_trajectory[index] + np.inner(coeff,previous_prediction)*weight) / (1+weight)
    elif method is "no_stack":
        alpha = 1/(1+smooth_weight)
        beta = beta_trend
        roll_trajectory = np.zeros(human.shape)
        filtered_trajectory = np.zeros(human.shape)
        filtered_res = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+2*tao):
                roll_trajectory[index] = human[index]
                filtered_trajectory[index] = human[index]
                filtered_res[index] = 0
            for index in np.arange(item[0]+2*tao, item[1]+1):
                feature = autoreg_game_context[index,:] # no stacking
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                previous_res = filtered_res[index-tao:index].copy()
                previous_res = previous_res[::-1]
                roll_trajectory[index] = policy.predict(feature)
                #filtered_trajectory[index] = (roll_trajectory[index] + np.inner(coeff,previous_prediction)*weight) / (1+weight)
                filtered_trajectory[index] = alpha*roll_trajectory[index] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                filtered_res[index] = beta*(filtered_trajectory[index] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res)
    elif method is "stack_pos_res": # assume the coefficients to be arranged in the order (position_smooth_coeff, residual_smooth_coeff)
        alpha = 1.0/(1+smooth_weight)
        beta = beta_trend
        roll_trajectory = np.zeros(human.shape)
        filtered_trajectory = np.zeros(human.shape)
        filtered_res = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+2*tao):
                roll_trajectory[index] = human[index]
                filtered_trajectory[index] = human[index]
                filtered_res[index] = 0
            for index in np.arange(item[0]+2*tao, item[1]+1):
                feature = autoreg_game_context[index,:]
                # stack position features
                for i in range(tao):
                    feature = np.append(feature,roll_trajectory[index-(i+1)])
                # stack residual features
                for i in range(tao):
                    previous_position = roll_trajectory[index-(i+1)-tao:index-(i+1)]
                    previous_position = previous_position[::-1]
                    feature = np.append(feature, roll_trajectory[index-(i+1)] - np.inner(coeff[:tao],previous_position)) #??? shouldn't be coeff[tao:]
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                previous_res = filtered_res[index-tao:index].copy()
                previous_res = previous_res[::-1]
                roll_trajectory[index] = policy.predict(feature)
                #filtered_trajectory[index] = (roll_trajectory[index] + np.inner(coeff,previous_prediction)*weight) / (1+weight)
                filtered_trajectory[index] = alpha*roll_trajectory[index] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                filtered_res[index] = beta*(filtered_trajectory[index] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res)
    elif method is "stack_pos_res_proper": # assume the coefficients to be arranged in the order (position_smooth_coeff, residual_smooth_coeff)
        alpha = 1.0/(1+smooth_weight)
        beta = beta_trend
        roll_trajectory = np.zeros(human.shape)
        filtered_trajectory = np.zeros(human.shape)
        filtered_res = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+2*tao):
                roll_trajectory[index] = human[index]
                filtered_trajectory[index] = human[index]
                filtered_res[index] = 0
            for index in np.arange(item[0]+tao, item[0]+2*tao):
                prev = filtered_trajectory[index-tao:index]
                prev = prev[::-1]
                filtered_res[index] = filtered_trajectory[index] - np.inner(coeff[0:tao], prev)
            for index in np.arange(item[0]+2*tao, item[1]+1):
                feature = autoreg_game_context[index,:]
                # stack position features
                for i in range(tao):
                    feature = np.append(feature,filtered_trajectory[index-(i+1)]) # use filtered trajectory to roll 
                # stack residual features
                for i in range(tao):
                    #previous_position = filtered_trajectory[index-(i+1)-tao:index-(i+1)]
                    #previous_position = previous_position[::-1]
                    #feature = np.append(feature, filtered_trajectory[index-(i+1)] - np.inner(coeff[:tao],previous_position)) #use filtred trajectory to roll
                    feature = np.append(feature, filtered_res[index-(i+1)]) # use filtered residuals as velocity 
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                previous_res = filtered_res[index-tao:index].copy()
                previous_res = previous_res[::-1]
                roll_trajectory[index] = policy.predict(feature)
                #filtered_trajectory[index] = (roll_trajectory[index] + np.inner(coeff,previous_prediction)*weight) / (1+weight)
                filtered_trajectory[index] = alpha*roll_trajectory[index] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                filtered_res[index] = beta*(filtered_trajectory[index] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res) ##Think about this carefully
    elif method is "stack_pos_res_predict": # assume the coefficients to be arranged in the order (position_smooth_coeff, residual_smooth_coeff)
        alpha = 1.0/(1+smooth_weight)
        beta = beta_trend
        roll_trajectory = np.zeros(human.shape)
        filtered_trajectory = np.zeros(human.shape)
        filtered_res = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+2*tao):
                roll_trajectory[index] = human[index]
                filtered_trajectory[index] = human[index]
                filtered_res[index] = 0
            for index in np.arange(item[0]+2*tao, item[1]+1):
                feature = autoreg_game_context[index,:]
                # stack position features
                for i in range(tao):
                    feature = np.append(feature,filtered_trajectory[index-(i+1)]) # use filtered trajectory to roll 
                # stack residual features
                for i in range(tao):
                    previous_position = filtered_trajectory[index-(i+1)-tao:index-(i+1)] # to consider: whether use filter or roll here to calculate residual features ???
                    previous_position = previous_position[::-1]
                    feature = np.append(feature, roll_trajectory[index-(i+1)] - np.inner(coeff[:tao],previous_position)) #use roll to predict velocity
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                previous_res = filtered_res[index-tao:index].copy()
                previous_res = previous_res[::-1]
                roll_trajectory[index] = policy.predict(feature)
                #filtered_trajectory[index] = (roll_trajectory[index] + np.inner(coeff,previous_prediction)*weight) / (1+weight)
                filtered_trajectory[index] = alpha*roll_trajectory[index] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                filtered_res[index] = beta*(filtered_trajectory[index] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res)
    elif method is "predict_pos_res": # assume the coefficients to be arranged in the order (position_smooth_coeff, residual_smooth_coeff)
        alpha = 1/(1+smooth_weight)
        beta = beta_trend
        roll_trajectory = np.zeros(human.shape)
        filtered_trajectory = np.zeros(human.shape)
        filtered_res = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao+tao):
                roll_trajectory[index] = human[index]
                filtered_trajectory[index] = human[index]
                filtered_res[index] = 0
            for index in np.arange(item[0]+tao+tao, item[1]+1):
                feature = autoreg_game_context[index,:]
                # stack position features
                for i in range(tao):
                    feature = np.append(feature,roll_trajectory[index-(i+1)])
                # stack residual features
                for i in range(tao):
                    previous_position = roll_trajectory[index-(i+1)-tao:index-(i+1)]
                    previous_position = previous_position[::-1]
                    feature = np.append(feature, (roll_trajectory[index-(i+1)] - np.inner(coeff[0:tao],previous_position)))
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                previous_res = filtered_res[index-tao:index].copy()
                previous_res = previous_res[::-1]
                roll_trajectory[index] = policy.predict(feature)[0,0] #predicted position
                #filtered_trajectory[index] = (roll_trajectory[index] + np.inner(coeff,previous_prediction)*weight) / (1+weight)
                filtered_trajectory[index] = alpha*roll_trajectory[index] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                #filtered_res[index] = beta*(filtered_trajectory[index] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res) #original double exp smooth
                filtered_res[index] = beta*(policy.predict(feature)[0,1]) + (1-beta)*np.inner(coeff[tao:], previous_res) # use predicted residual here
                #filtered_res[index] = beta*(roll_trajectory[index] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res)
    elif method is "stack_pos":
        alpha = 1/(1+smooth_weight)
        beta = beta_trend        
        roll_trajectory = np.zeros(human.shape)
        filtered_trajectory = np.zeros(human.shape)
        filtered_res = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                roll_trajectory[index] = human[index]
                filtered_trajectory[index] = human[index]
                filtered_res[index] = 0
            for index in np.arange(item[0]+tao, item[1]+1):
                feature = autoreg_game_context[index,:]
                for i in range(tao):
                    feature = np.append(feature,roll_trajectory[index-(i+1)])
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                roll_trajectory[index] = policy.predict(feature)
                #filtered_trajectory[index] = (roll_trajectory[index] + np.inner(coeff,previous_prediction)*weight) / (1+weight)       
                filtered_trajectory[index] = alpha*roll_trajectory[index] + (1-alpha)*(np.inner(coeff, previous_prediction) + filtered_res[index-1])
                filtered_res[index] = beta*(filtered_trajectory[index] - np.inner(coeff, previous_prediction)) + (1-beta)*filtered_res[index-1] 
    elif method is "stack_pos_proper":
        alpha = 1/(1+smooth_weight)
        beta = beta_trend        
        roll_trajectory = np.zeros(human.shape)
        filtered_trajectory = np.zeros(human.shape)
        filtered_res = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                roll_trajectory[index] = human[index]
                filtered_trajectory[index] = human[index]
                filtered_res[index] = 0
            for index in np.arange(item[0]+tao, item[1]+1):
                feature = autoreg_game_context[index,:]
                for i in range(tao):
                    feature = np.append(feature,filtered_trajectory[index-(i+1)]) # this is the main difference between stack_pos and stack_pos_proper
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                roll_trajectory[index] = policy.predict(feature)
                #filtered_trajectory[index] = (roll_trajectory[index] + np.inner(coeff,previous_prediction)*weight) / (1+weight)       
                filtered_trajectory[index] = alpha*roll_trajectory[index] + (1-alpha)*(np.inner(coeff, previous_prediction) + filtered_res[index-1])
                filtered_res[index] = beta*(filtered_trajectory[index] - np.inner(coeff, previous_prediction)) + (1-beta)*filtered_res[index-1]     
    return filtered_trajectory

def roll_and_smooth_test_trajectory(policy, reference_path, context, coeff, weight, method):
    if method is "stack_pos_res":
        alpha = 1.0/(1+weight)
        beta = beta_trend        
        Y_predict = np.zeros(reference_path.shape)
        Y_roll = np.zeros(reference_path.shape)
        Y_res = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<2*tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                Y_roll[i] = reference_path[i]
                Y_res[i] = 0
            else:
                feature = context[i]
                #stack position features
                for j in range(tao):
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]))
                    feature = np.append(feature, Y_roll[i-(j+1)])
                #stack residual features
                for j in range(tao):
                    previous_position = Y_roll[i-(j+1)-tao:i-(j+1)]
                    previous_position = previous_position[::-1]
                    feature = np.append(feature, Y_roll[i-(j+1)] - np.inner(coeff[0:tao], previous_position))
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]-Y_roll[i-(j+2)]))

                previous_prediction = Y_predict[i-tao:i].copy()
                previous_prediction = previous_prediction[::-1]
                previous_res = Y_res[i-tao:i].copy()
                previous_res = previous_res[::-1]
                Y_roll[i] = policy.predict(feature)[0]
                Y_predict[i] = alpha*Y_roll[i] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                Y_res[i] = beta*(Y_predict[i] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res)
                #Y_predict[i] = (Y_roll[i] + np.inner(coeff, previous_prediction)*weight) / (1+weight)
                #Y_predict[i] = (policy.predict(feature)[0] + np.inner(coeff,previous_prediction)*weight)/(1+weight) # replace
    elif method is "stack_pos_res_proper":
        alpha = 1.0/(1+weight)
        beta = beta_trend        
        Y_predict = np.zeros(reference_path.shape)
        Y_roll = np.zeros(reference_path.shape)
        Y_res = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                Y_roll[i] = reference_path[i]
                Y_res[i] = 0
            elif i < 2*tao:
                Y_predict[i] = reference_path[i]
                Y_roll[i] = reference_path[i]
                prev = Y_predict[i-tao:i]
                prev = prev[::-1]
                Y_res[i] = Y_predict[i] - np.inner(coeff[0:tao], prev)
            else:
                feature = context[i]
                #stack position features
                for j in range(tao):
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]))
                    feature = np.append(feature, Y_predict[i-(j+1)]) #use filtered trajectory to roll
                #stack residual features
                for j in range(tao):
                    #previous_position = Y_predict[i-(j+1)-tao:i-(j+1)] #use filtered trajectory to roll
                    #previous_position = previous_position[::-1]
                    #feature = np.append(feature, Y_predict[i-(j+1)] - np.inner(coeff[0:tao], previous_position)) #use filter to roll
                    feature = np.append(feature, Y_res[i-(j+1)])

                previous_prediction = Y_predict[i-tao:i].copy()
                previous_prediction = previous_prediction[::-1]
                previous_res = Y_res[i-tao:i].copy()
                previous_res = previous_res[::-1]
                Y_roll[i] = policy.predict(feature)[0]
                Y_predict[i] = alpha*Y_roll[i] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                Y_res[i] = beta*(Y_predict[i] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res)
    elif method is "stack_pos_res_predict":
        alpha = 1.0/(1+weight)
        beta = beta_trend        
        Y_predict = np.zeros(reference_path.shape)
        Y_roll = np.zeros(reference_path.shape)
        Y_res = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<2*tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                Y_roll[i] = reference_path[i]
                Y_res[i] = 0
            else:
                feature = context[i]
                #stack position features
                for j in range(tao):
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]))
                    feature = np.append(feature, Y_predict[i-(j+1)]) #use filtered trajectory to roll
                #stack residual features
                for j in range(tao):
                    previous_position = Y_predict[i-(j+1)-tao:i-(j+1)] ## to consider: whether use filter or roll here to calculate residual features ???
                    previous_position = previous_position[::-1]
                    feature = np.append(feature, Y_roll[i-(j+1)] - np.inner(coeff[0:tao], previous_position)) #use filter to roll
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]-Y_roll[i-(j+2)]))

                previous_prediction = Y_predict[i-tao:i].copy()
                previous_prediction = previous_prediction[::-1]
                previous_res = Y_res[i-tao:i].copy()
                previous_res = previous_res[::-1]
                Y_roll[i] = policy.predict(feature)[0]
                Y_predict[i] = alpha*Y_roll[i] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                Y_res[i] = beta*(Y_predict[i] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res)
    elif method is "predict_pos_res":
        alpha = 1.0/(1+smooth_weight)
        beta = beta_trend        
        Y_predict = np.zeros(reference_path.shape)
        Y_roll = np.zeros(reference_path.shape)
        Y_res = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<(tao+tao):
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                Y_roll[i] = reference_path[i]
                Y_res[i] = 0
            else:
                feature = context[i]
                #stack position features
                for j in range(tao):
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]))
                    feature = np.append(feature, Y_roll[i-(j+1)])
                #stack residual features
                for j in range(tao):
                    previous_position = Y_roll[i-(j+1)-tao:i-(j+1)]
                    previous_position = previous_position[::-1]
                    feature = np.append(feature, (Y_roll[i-(j+1)] - np.inner(coeff[0:tao], previous_position)))
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]-Y_roll[i-(j+2)]))

                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                previous_res = Y_res[i-tao:i]
                previous_res = previous_res[::-1]
                Y_roll[i] = policy.predict(feature)[0,0] 
                Y_predict[i] = alpha*Y_roll[i] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                #Y_res[i] = beta*(Y_predict[i] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res) #original double exp smoothing
                Y_res[i] = beta*(policy.predict(feature)[0,1]) + (1-beta)*np.inner(coeff[tao:], previous_res) #use predicted residual here
                #Y_res[i] = beta*(Y_roll[i] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res)
    elif method is "no_stack":
        alpha = 1/(1+smooth_weight)
        beta = beta_trend        
        Y_predict = np.zeros(reference_path.shape)
        Y_roll = np.zeros(reference_path.shape)
        Y_res = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<2*tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                Y_roll[i] = reference_path[i]
                Y_res[i] = 0
            else:
                feature = context[i]
                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                previous_res = Y_res[i-tao:i]
                previous_res = previous_res[::-1]
                Y_roll[i] = policy.predict(feature)[0]
                Y_predict[i] = alpha*Y_roll[i] + (1-alpha)* (np.inner(coeff[0:tao],previous_prediction)+ np.inner(coeff[tao:], previous_res) )
                Y_res[i] = beta*(Y_predict[i] - np.inner(coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(coeff[tao:], previous_res)        
    elif method is "stack_vel_pos":
        Y_predict = np.zeros(reference_path.shape)
        Y_roll = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                Y_roll[i] = reference_path[i]
            else:
                feature = context[i]
                for j in range(tao-1):
                    feature = np.hstack((feature,Y_roll[i-(j+1)]-Y_roll[i-(j+2)]))
                for j in range(tao):
                    feature = np.hstack((feature,Y_roll[i-(j+1)]))
                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                Y_roll[i] = policy.predict(feature)[0]
                Y_predict[i] = (Y_roll[i] + np.inner(coeff, previous_prediction)*weight) / (1+weight)
                #Y_predict[i] = (policy.predict(feature)[0] + np.inner(coeff,previous_prediction)*weight)/(1+weight) # replace
    elif method is "stack_pos":
        alpha = 1.0/(1+weight)
        beta = beta_trend                
        Y_predict = np.zeros(reference_path.shape)
        Y_roll = np.zeros(reference_path.shape)
        Y_res = np.zeros(reference_path.shape) #add for double exponential smoothing
        for i in range(len(reference_path)):
            if i<tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                Y_roll[i] = reference_path[i]
                Y_res[i] = 0
            else:
                feature = context[i]
                for j in range(tao):
                    feature = np.hstack((feature,Y_roll[i-(j+1)]))
                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                Y_roll[i] = policy.predict(feature)[0]
                Y_predict[i] = (Y_roll[i] + np.inner(coeff, previous_prediction)*weight) / (1+weight)    
                Y_predict[i] = alpha*Y_roll[i] + (1-alpha)*(np.inner(coeff, previous_prediction) +Y_res[i-1])
                Y_res[i] = beta*(Y_predict[i] - np.inner(coeff, previous_prediction)) + (1-beta) * Y_res[i-1]   
    elif method is "stack_pos_proper":
        alpha = 1.0/(1+weight)
        beta = beta_trend                
        Y_predict = np.zeros(reference_path.shape)
        Y_roll = np.zeros(reference_path.shape)
        Y_res = np.zeros(reference_path.shape) #add for double exponential smoothing
        for i in range(len(reference_path)):
            if i<tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                Y_roll[i] = reference_path[i]
                Y_res[i] = 0
            else:
                feature = context[i]
                for j in range(tao):
                    feature = np.hstack((feature,Y_predict[i-(j+1)])) #main difference between stack_pos and stack_pos_proper
                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                Y_roll[i] = policy.predict(feature)[0]
                Y_predict[i] = (Y_roll[i] + np.inner(coeff, previous_prediction)*weight) / (1+weight)    
                Y_predict[i] = alpha*Y_roll[i] + (1-alpha)*(np.inner(coeff, previous_prediction) +Y_res[i-1])
                Y_res[i] = beta*(Y_predict[i] - np.inner(coeff, previous_prediction)) + (1-beta) * Y_res[i-1]   
    return Y_predict

def interpolate_and_smooth_learned_policy(old_policy, new_policy, interpolate, old_coeff, new_coeff, weight, method):
    if method is "stack_pos_res":
        alpha = 1/(1+weight)
        beta = beta_trend
        filtered_trajectory = np.zeros(human.shape)
        old_trajectory = np.zeros(human.shape)
        new_trajectory = np.zeros(human.shape)
        filtered_res = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0], item[0]+2*tao):
                old_trajectory[index] = human[index]
                new_trajectory[index] = human[index]
                filtered_trajectory[index] = human[index]
                filtered_res[index] = 0
            for index in np.arange(item[0]+2*tao, item[1]+1):
                old_feature = autoreg_game_context[index,:]
                new_feature = autoreg_game_context[index,:]
                for i in range(tao):
                    old_feature = np.append(old_feature, old_trajectory[index-(i+1)])
                    new_feature = np.append(new_feature, new_trajectory[index-(i+1)])
                for i in range(tao):
                    old_previous_position = old_trajectory[index-(i+1)-tao:index-(i+1)]
                    old_previous_position = old_previous_position[::-1]
                    old_feature = np.append(old_feature, old_trajectory[index-(i+1)] - np.inner(old_coeff[:tao],old_previous_position))
                    new_previous_position = new_trajectory[index-(i+1)-tao:index-(i+1)]
                    new_previous_position = new_previous_position[::-1]
                    new_feature = np.append(new_feature, new_trajectory[index-(i+1)] - np.inner(new_coeff[:tao],new_previous_position))                    
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                old_trajectory[index] = old_policy.predict(old_feature)
                new_trajectory[index] = new_policy.predict(new_feature)
                
                previous_res = filtered_res[index-tao:index].copy()
                previous_res = previous_res[::-1]
                new_prediction = interpolate * new_trajectory[index] + (1-interpolate)* old_trajectory[index]
                filtered_trajectory[index] = alpha*new_prediction + (1-alpha)* (np.inner(new_coeff[0:tao],previous_prediction)+ np.inner(new_coeff[tao:], previous_res))
                filtered_res[index] = beta*(filtered_trajectory[index] - np.inner(new_coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(new_coeff[tao:], previous_res)
    elif method is "stack_vel_pos":
        filtered_trajectory = np.zeros(human.shape)
        old_trajectory = np.zeros(human.shape)
        new_trajectory = np.zeros(human.shape)
        for item in inPlay:
            for index in np.arange(item[0],item[0]+tao):
                filtered_trajectory[index] = human[index]
                old_trajectory[index] = human[index]
                new_trajectory[index] = human[index]
            for index in np.arange(item[0]+tao,item[1]+1):
                old_feature = autoreg_game_context[index,:]
                new_feature = autoreg_game_context[index,:]
                for i in range(tao-1):
                    old_feature = np.append(old_feature, old_trajectory[index-(i+1)] - old_trajectory[index-(i+2)])
                    new_feature = np.append(new_feature, new_trajectory[index-(i+1)] - new_trajectory[index-(i+2)])
                for i in range(tao):
                    old_feature = np.append(old_feature,old_trajectory[index-(i+1)])
                    new_feature = np.append(new_feature,new_trajectory[index-(i+1)])
                previous_prediction = filtered_trajectory[index-tao:index].copy()
                previous_prediction = previous_prediction[::-1]
                old_trajectory[index] = old_policy.predict(old_feature)
                new_trajectory[index] = new_policy.predict(new_feature)

                #old_model_predict = (old_policy.predict(feature) + np.inner(old_coeff, previous_prediction) * weight) / (1+weight)
                #new_model_predict = (new_policy.predict(feature) + np.inner(new_coeff, previous_prediction) * weight) / (1+weight)
                #current_prediction = interpolate * new_policy.predict(feature) + (1-interpolate) * old_policy.predict(feature)
                filtered_trajectory[index] = (interpolate * new_trajectory[index] + (1-interpolate) * old_trajectory[index] + np.inner(new_coeff, previous_prediction) * weight) / (1+weight)
    return filtered_trajectory 

def interpolate_and_smooth_test_policy(old_policy, new_policy, interpolate, reference_path, context, old_coeff, new_coeff, weight, method):
    if method is "stack_pos_res":
        alpha = 1/(1+weight)
        beta = beta_trend        
        Y_predict = np.zeros(reference_path.shape)
        old_roll = np.zeros(reference_path.shape)
        new_roll = np.zeros(reference_path.shape)
        Y_res = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<2*tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                old_roll[i] = reference_path[i]
                new_roll[i] = reference_path[i]
                Y_res[i] = 0
            else:
                old_feature = context[i]
                new_feature = context[i]
                #stack position features
                for j in range(tao):
                    #feature = np.hstack((feature,Y_roll[i-(j+1)]))
                    old_feature = np.append(old_feature, old_roll[i-(j+1)])
                    new_feature = np.append(new_feature, new_roll[i-(j+1)])
                #stack residual features
                for j in range(tao):
                    old_previous_position = old_roll[i-(j+1)-tao:i-(j+1)]
                    old_previous_position = old_previous_position[::-1]
                    old_feature = np.append(old_feature, old_roll[i-(j+1)] - np.inner(old_coeff[0:tao], old_previous_position))
                    new_previous_position = new_roll[i-(j+1)-tao:i-(j+1)]
                    new_previous_position = new_previous_position[::-1]
                    new_feature = np.append(new_feature, new_roll[i-(j+1)] - np.inner(new_coeff[0:tao], new_previous_position))
                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                old_roll[i] = old_policy.predict(old_feature)
                new_roll[i] = new_policy.predict(new_feature)
                
                previous_res = Y_res[i-tao:i].copy()
                previous_res = previous_res[::-1]
                new_prediction = interpolate * new_roll[i] + (1-interpolate)* old_roll[i]

                #Y_roll[i] = policy.predict(feature)[0]
                Y_predict[i] = alpha*new_prediction + (1-alpha)* (np.inner(new_coeff[0:tao],previous_prediction)+ np.inner(new_coeff[tao:], previous_res))
                Y_res[i] = beta*(Y_predict[i] - np.inner(new_coeff[0:tao], previous_prediction)) + (1-beta)*np.inner(new_coeff[tao:], previous_res)        

    elif method is "stack_vel_pos":
        Y_predict = np.zeros(reference_path.shape)
        old_roll = np.zeros(reference_path.shape)
        new_roll = np.zeros(reference_path.shape)
        for i in range(len(reference_path)):
            if i<tao:
                Y_predict[i] = reference_path[i] #note: have the first tau frames correct
                old_roll[i] = reference_path[i]
                new_roll[i] = reference_path[i]
            else:
                old_feature = context[i]
                new_feature = context[i]
                for j in range(tao-1):
                    old_feature = np.hstack((old_feature,old_roll[i-(j+1)]-old_roll[i-(j+2)]))
                    new_feature = np.hstack((new_feature,new_roll[i-(j+1)]-new_roll[i-(j+2)]))
                for j in range(tao):
                    old_feature = np.hstack((old_feature,old_roll[i-(j+1)]))
                    new_feature = np.hstack((new_feature,new_roll[i-(j+1)]))
                previous_prediction = Y_predict[i-tao:i]
                previous_prediction = previous_prediction[::-1]
                old_roll[i] = old_policy.predict(old_feature)[0]
                new_roll[i] = new_policy.predict(new_feature)[0]
                #current_prediction = interpolate * new_policy.predict(feature) + (1-interpolate) * old_policy.predict(feature)
                #old_model_predict = (old_policy.predict(feature) + np.inner(old_coeff, previous_prediction) * weight) / (1+weight)
                #new_model_predict = (new_policy.predict(feature) + np.inner(new_coeff, previous_prediction) * weight) / (1+weight)
                #Y_predict[i] = (current_prediction + np.inner(coeff,previous_prediction)*weight)/(1+weight) # replace
                Y_predict[i] = (interpolate * new_roll[i] + (1-interpolate) * old_roll[i] + np.inner(new_coeff, previous_prediction) * weight) / (1+weight)
    return Y_predict       