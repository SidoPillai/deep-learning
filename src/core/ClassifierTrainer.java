import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


public class ClassifierTrainer {

	/**
	 * Trains the Convolutional neural network.
	 * @param X numjava object which contains training samples
	 * @param y numjava object which contains class scores for training samples
	 * @param XVal numjava object which contains validation samples
	 * @param yVal numjava object which contains class scores for validation samples
	 * @param model Model parameters weights and biases
	 * @param reg regularization factor
	 * @param learningRate  learning rate used for optimization of network 
	 * @param momentum momentum used for optimization of network
	 * @param learningRateDecay 
	 * @param update Whether to use SGD,momentum,Nestrov's optimization technique.
	 * @param sampleBatches To check if training is to be done using a mini-batch.
	 * @param numEpochs Total number of epochs 
	 * @param batchSize Batch size to be used.
	 * @param filterHeight Size of the filter
	 * @param orgimgWidth Original image width
	 * @param orgimgHeight Original image height.
	 * 
	 * @return Map which contains best model parameters, loss history, training accuracy history and validation history.
	 * @throws Exception
	 */
	public  Map<String,Object> train (numjava X, numjava y, numjava XVal, numjava yVal, Map<String,Object>model, float reg,
			float learningRate, float momentum, float learningRateDecay, String update ,boolean sampleBatches,
			int numEpochs, int batchSize, int filterHeight,int orgimgWidth, int orgimgHeight) throws Exception
			{
			
			ConvoNet convNet = new ConvoNet();
			
			int M = X.M;
			int iterationsPerEpoch;
			int noOfiters = 0 ;
			int epochs = 0;
			float bestValAcc = 0;
			// To store best model parameters.
			Map<String, Object> bestModel = new HashMap<String, Object>();
			ArrayList<Float> lossHistory = new ArrayList<Float>();  // Store loss, so that we can plot.  
			//To store train accuracy.
			ArrayList<Float> trainAccHistory =  new ArrayList<Float>();
			//To store validation accuracy.
			ArrayList<Float> valAccHistory = new ArrayList<Float>();
			// Used for trainig the network
			numjava iterArray_X = null;
			numjava iterArray_y = null;
			//To store random arrays for training accuracy
			numjava trainArray_X = new numjava(1000,X.N);
			numjava trainArray_y = new numjava(1000,y.N);
			// Array to store random numbers, these are then used for training.
			int [] randArray = new int [1000];
			if (sampleBatches)
			{
				iterationsPerEpoch = M/batchSize;
				iterArray_X =  new numjava(batchSize,X.N);
				iterArray_y = new numjava(1,batchSize);
			}
			else 
			{
				iterationsPerEpoch = 1;  // SGD
	
	
			}
	
			noOfiters = iterationsPerEpoch*numEpochs;
	
			for (int it = 0; it < noOfiters; it++)
			{
				if(it%10==0){
					System.out.println("Starting Iteration	"+it);
				}	
	
				// Get batches, SGD
				if(sampleBatches)
				{
					// Get an array in iterArray with size equal to batch size.iterArray has values of random images. 
					numjava.randomMask(M, batchSize, X, iterArray_X);
					numjava.randomMask(M, batchSize, y, iterArray_y);
	
				}
				// For Full Batch, No SGD
				else
				{
					iterArray_X = X;
					iterArray_y = y;	
				}
	
				// Evaluating the loss 
				Map<String,Object> lossgrad = new HashMap<String,Object>(); 
				lossgrad = convNet.layerConvonet(iterArray_X, model, iterArray_y, reg, filterHeight, orgimgWidth, orgimgHeight,false);
				lossHistory.add((Float) lossgrad.get("loss"));
				lossgrad.remove("loss");
				numjava dx;
				String key;
				numjava modelval;
				for (Map.Entry<String, Object> entry : lossgrad.entrySet())
				{
					key = entry.getKey();
					dx = (numjava)entry.getValue();
					dx = numjava.MulbyVal(dx, (-1*learningRate));
					modelval = (numjava)model.get(key);
					modelval = numjava.add(modelval, dx);
					model.remove(key);
					model.put(key,modelval); // update with new value w1,w2,b1,b2
				}
	
	
	
				if(M>1000)
				{
					for (int j=0; j < 1000; j++)
					{
						// creates an array with random points, so that it can be used for getting random images for training.
						numjava.getrandom(M, batchSize, X, randArray);
						trainArray_X.finalmatrix[j] = X.finalmatrix[randArray[j]];
						trainArray_y.finalmatrix[j] = y.finalmatrix[randArray[j]];
					}
	
				}
				else
				{
					trainArray_X = X;
					trainArray_y = y;
				}
	
				//Evaluating training accuracy
				numjava scores = (numjava) convNet.layerConvonet(trainArray_X, model, trainArray_y, reg, filterHeight, orgimgWidth, orgimgHeight, true);
				float [] yPredTrain = numjava.get_Arg_Max(scores);
				float mean = numjava.getMean(yPredTrain, y.finalmatrix[0]);
				trainAccHistory.add(mean);
	
				// Evaluating validation accuracy.
				numjava scoresVal = (numjava) convNet.layerConvonet(XVal, model, yVal, reg, filterHeight, orgimgWidth, orgimgHeight, true);
				float [] yPredVal = numjava.get_Arg_Max(scoresVal);
				float meanVal = numjava.getMean(yPredVal, yVal.finalmatrix[0]);
				valAccHistory.add(meanVal);
				String keyVal;
				boolean flag = true;
				// Get the best model parameters.
				
				if(meanVal>bestValAcc)
				{
					bestValAcc = meanVal;
					if(flag)
					{   // For the 1st time just put values
						for(Map.Entry<String, Object> entry: model.entrySet())
						{   
							keyVal = entry.getKey();
							bestModel.put(keyVal, numjava.deepCopy((numjava)model.get(keyVal)));
							flag = false;
						}
					}
					else
					{
						// Now remove the previous values from the bestModel and add new values.
						for(Map.Entry<String, Object> entry: model.entrySet())
						{   
							keyVal = entry.getKey();
							bestModel.remove(keyVal);
							bestModel.put(keyVal, numjava.deepCopy((numjava)model.get(keyVal)));
						}
					}
				}
	
				System.out.print("Best validation accuracy"+bestValAcc);
	
	
				// Incomplete.	
	
			}
	
			Map<String,Object> ret = new HashMap<String,Object>();
			ret.put("bestModel",bestModel);
			ret.put("lossHistory", lossHistory);
			ret.put("trainAccHistory", trainAccHistory);
			ret.put("valAccHistory",valAccHistory);
	
			return ret;
			
			}

}
