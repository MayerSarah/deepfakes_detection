from pytest import raises

#test dataset non null
def test_data_shape():
    awaited_result: shape =! (0,0)
    assert load_data().shape == awaited_result

#test our model output for the train is size [batch_size, num_classes]
def output_shape_test():
    awaited_result : numpy_array = np.ones([batch_size, target_size])
    assert  model_output == awaited_result

#test accuracy of the model is ok 
def performances_check():
    awaited_result : accuracy > 0.6
    assert model_accuracy == awaited_result 

#test output for the prediction dataset is the goodsize
def prediction_shape():
    awaited_result : shape = (3679, 1)
    assert len(prediction) = awaited_result
