import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * The class helps in training Convolutional Neural Network.
 * @author suhaspillai
 *
 */

public class ClassifierTrainer {

	/**
	 * The method trains the Convolutional neural network.
	 * @param X numjava object which contains training samples
	 * @param y numjava object which contains class scores for training samples
	 * @param model Model parameters weights and biases
	 * @param reg regularization factor
	 * @param learningRate  learning rate used for optimization of network 
	 * @param momentum momentum used for optimization of network
	 * @param learningRateDecay 
	 * @param sampleBatches To check if training is to be done using a mini-batch.
	 * @param numEpochs Total number of epochs 
	 * @param batchSize Batch size to be used.
	 *
	 * @return Map which contains best model parameters, loss history, training accuracy history and validation history.
	 * @throws Exception
	 */

	public  Map<String,Object> train (numjava xTrain, numjava yTrain, numjava xVal, numjava yVal, Map<String,Object>model, double reg,
			double learningRate, double momentum,boolean sampleBatches,
			int numEpochs, int batchSize, Map<String,Integer>convParam) throws Exception
			{
		BufferedWriter wr = new BufferedWriter(new FileWriter("accuray.txt"));
		ConvoNet convNet = new ConvoNet();
		int M = xTrain.M;
		int iterationsPerEpoch;
		int noOfiters = 0 ;
		int epochs = 0;
		double bestValAcc = 0;
		int pad = convParam.get("pad");
		int orgimgWidth = convParam.get("imageWidth");
		int orgimgHeight = convParam.get("imageHeight");
		int filterHeight = convParam.get("filterHeight");
		// To store best model parameters.
		Map<String, Object> bestModel = new HashMap<String, Object>();
		ArrayList<Double> lossHistory = new ArrayList<Double>();  // Store loss, so that we can plot.  
		//To store train accuracy.
		ArrayList<Double> trainAccHistory =  new ArrayList<Double>();
		//To store validation accuracy.
		ArrayList<Double> valAccHistory = new ArrayList<Double>();
		// Used for trainig the network
		numjava iterArray_X = null;
		numjava iterArray_y = null;
		//To store random arrays for training accuracy
		numjava trainArray_X =null;
		numjava trainArray_y=null;
		// Array to store random numbers, these are then used for training.
		int [] randArray = new int [1000];
		int [] randBatch = new int[batchSize];
		int randCount=0;
		double learningRateDecay = 0.95;
		int accuracy_var = 50;	
		boolean momentumflag = true;

		Map<String,Object> step_cache  = null;
		if (sampleBatches)
		{	System.out.println("inside smaple batches.");
		iterationsPerEpoch = M/batchSize;

		}
		else 
		{
			iterationsPerEpoch = 1;  // SGD

		}

		noOfiters = iterationsPerEpoch*numEpochs;
		boolean flag = true;

		for (int it = 0; it < noOfiters; it++)
		{
			if(it%10==0){
				System.out.println("Starting Iteration	"+it);
			}	

			// Get batches, SGD
			if(sampleBatches)
			{
				//					System.out.println("Inside sample batches.");
				iterArray_X =  new numjava(batchSize,xTrain.N);
				iterArray_y = new numjava(batchSize,1);
				// Get an array in iterArray with size equal to batch size.iterArray has values of random images. 
				//	System.out.println("Get values for smaple batches.");
				numjava.getrandom(M, batchSize, randBatch);

				for(int i = 0 ; i < batchSize;i++)
				{
					//numjava.randomMask(M, batchSize, X, iterArray_X);
					//numjava.randomMask(M, batchSize, y, iterArray_y);
					randCount = randBatch[i];
					iterArray_X.finalmatrix[i] = Arrays.copyOf(xTrain.finalmatrix[randCount],xTrain.finalmatrix[i].length);
					iterArray_y.finalmatrix[i][0]= yTrain.finalmatrix[randCount][0];
				}
			}
			// For Full Batch, No SGD
			else

			{
				//					System.out.println("Full Batch");
				iterArray_X = xTrain;
				iterArray_y = yTrain;	
			}

			Map<String,Object> lossgrad;	
			lossgrad = convNet.layerConvonet(iterArray_X, model, iterArray_y, reg,false,convParam);
			lossHistory.add((double) lossgrad.get("loss"));
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


			// for momentum update	
			/*
				for (Map.Entry<String, Object> entry : lossgrad.entrySet())
				{	
					if(momentumflag = true)  // only for the first time
					{

						step_cache  = new HashMap<String,Object>();
						key = entry.getKey();
						numjava grad = (numjava)entry.getValue(); // gradients calculated from backprop
						dx = new numjava(grad.M,grad.N); // for step cache
						step_cache.put(key,dx);
						dx =numjava.add(numjava.MulbyVal((numjava)step_cache.get(key),momentum), numjava.MulbyVal(grad, (-1*learningRate)));
						step_cache.remove(key);
						step_cache.put(key,dx);


					}
					else
					{
						key = entry.getKey();
						numjava grad = (numjava)entry.getValue(); // gradients calculated from backprop
						dx =numjava.add(numjava.MulbyVal((numjava)step_cache.get(key),momentum), numjava.MulbyVal(grad, (-1*learningRate)));
						step_cache.remove(key);
						step_cache.put(key,dx);

					}
					model.remove(key);
					model.put(key,dx);

				}
				momentumflag=false;	
			 */
			// Normal update

			// Change the learning rate after specific iterations,if training accuracy does not increase.
			if(((it+1)%iterationsPerEpoch==0)||(it%accuracy_var)==0)
			{
				learningRate = learningRate * learningRateDecay;
			}



			if(M>=1000)
			{
				//					System.out.println("Test random training accuracy");
				trainArray_X= new numjava(1000,xTrain.N);
				trainArray_y = new numjava(1000,yTrain.N);
				numjava.getrandom(M, 1000, randArray);
				for (int j=0; j < 1000; j++)
				{
					// creates an array with random points, so that it can be used for getting random images for training.
					randCount = randArray[j];
					trainArray_X.finalmatrix[j] = Arrays.copyOf(xTrain.finalmatrix[randCount],xTrain.finalmatrix[randCount].length);
					trainArray_y.finalmatrix[j][0] = yTrain.finalmatrix[randCount][0];
				}

			}
			else
			{
				//					System.out.println("Inside random train accuracy else part");	
				trainArray_X = xTrain;
				trainArray_y = yTrain;
			}

			//Evaluating training accuracy
			Map<String,Object> trainAcc= convNet.layerConvonet(trainArray_X, model, trainArray_y, reg, true,convParam);
			numjava scores = (numjava)trainAcc.get("scores");
			double [] yPredTrain = numjava.get_Arg_Max(scores);
			double mean = numjava.getMean(yPredTrain,trainArray_y);
			trainAccHistory.add(mean);
			System.out.println("Training Accuracy: "+mean);
			trainArray_X = null;
			trainArray_y = null;

			// Evaluating validation accuracy.
			Map<String,Object> trainVal = convNet.layerConvonet(xVal, model, yVal, reg, true,convParam);
			numjava scoresVal = (numjava)trainVal.get("scores");
			double [] yPredVal = numjava.get_Arg_Max(scoresVal);
			double meanVal = numjava.getMean(yPredVal, yVal);
			valAccHistory.add(meanVal);
			System.out.println("Validation accuracy:  "+meanVal);
			String keyVal;
			yPredVal=null;
			// Get the best model parameters.
			if(meanVal>bestValAcc)
			{
				bestValAcc = meanVal;
				if(flag)
				{   // For the 1st time just put values
					for(Map.Entry<String, Object> entry: model.entrySet())
					{   
						keyVal = entry.getKey();
						//							System.out.println("When meanVal is greater than best valacc: "+keyVal);
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
						//							System.out.println("When meanVal is greater than best valacc in Else: "+keyVal);
						bestModel.remove(keyVal);
						bestModel.put(keyVal, numjava.deepCopy((numjava)model.get(keyVal)));
					}
				}
				System.out.println("Best validation accuracy"+bestValAcc);
			}

		}
		xTrain = null;
		yTrain = null;
		xVal = null;
		yVal = null;
		wr.close();
		Map<String,Object> ret = new HashMap<String,Object>();
		ret.put("bestModel",bestModel);
		ret.put("lossHistory", lossHistory);
		ret.put("trainAccHistory", trainAccHistory);
		ret.put("valAccHistory",valAccHistory);
		ret.put("bestvalidationacc",bestValAcc);	
		return ret;

			}

}