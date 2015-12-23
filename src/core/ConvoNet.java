import java.util.HashMap;
import java.util.Map;



/**
 * The class ConvoNet contains method to train and load parameters for the network.
 * @author suhaspillai
 *
 */

public class ConvoNet {

	/**
	 * The method is used to call different architectures of convolutional neural network.
	 * @param x data for training convolutional neural network.
	 * @param model model contains parameters to train the network.
	 * @param y Data containing labels for the data
	 * @param reg Regularization parameter
	 * @param check If true this will just return scores.
	 * @param convParam Map containing values for convolutional layer.
	 * @return values for loss , weights and bias.
	 */

	public Map<String,Object> layerConvonet(numjava x, Map<String,Object> model, numjava y,double reg,boolean check,Map<String,Integer>convParam)
	{
		layer_utils layer_util = new layer_utils();
		layers layer = new layers();
		numjava w1 = (numjava)model.get("w1");
		numjava b1 = (numjava)model.get("b1");
		numjava w2 = (numjava)model.get("w2");
		numjava b2 = (numjava)model.get("b2");
		int M = x.M; // total no of images.

		int pad = convParam.get("pad");
		int orgimgWidth = convParam.get("imageWidth");
		int orgimgHeight = convParam.get("imageHeight");
		Map<String,Integer> poolParam = new HashMap<String,Integer>();
		poolParam.put("poolHeight",2);
		poolParam.put("poolWidth",2);
		poolParam.put("stride", 2);
		//Doing the convolution relu and pooling part.
		Map<String,Object>convReluPool = layer_util.convoReluPoolForward(x, w1, b1, convParam, poolParam, orgimgWidth, orgimgHeight);	
		// One forward propagation.
		numjava a1 = (numjava)convReluPool.get("out");
		Map<String,Object>affineForward = layer_util.affineForward(a1, w2, b2);

		if(check)
		{
			// This will return the scores which can then be used for training and cross validation.
			numjava scores = (numjava)affineForward.get("out");
			Map<String,Object> ret = new HashMap<String,Object>();
			ret.put("scores",scores);
			return ret;
		}

		// Now computing the soft max score
		// Here we get loss and the derivative of the soft max layer, which is used for back propagation. 
		Map<String,Object> softmaxout = layer.softmax_loss((numjava)affineForward.get("out"), y);
		double dataLoss =(double) softmaxout.get("loss");
		numjava dscores = (numjava)softmaxout.get("dx");

		//To compute the gradient using backward pass.
		//Sending affine forward to has out and cache obtained from affine forward. 
		Map<String,Object> affineBackward  = layer.affine_backward(dscores,affineForward);
		// Get derivative of w2,x, and b2 weight
		numjava dw2 = (numjava)affineBackward.get("dw");
		numjava dx2 = (numjava)affineBackward.get("dx");
		numjava db2  = (numjava)affineBackward.get("db");
		Map<String,Object> convReluPoolBackward = layer_util.convoReluPoolBackward(dx2, convReluPool, orgimgWidth, orgimgHeight);
		//get derivative for conv mrelu and max pool
		numjava dw1 = (numjava)convReluPoolBackward.get("dw");
		numjava dx1 = (numjava) convReluPoolBackward.get("dx");
		numjava db1 = (numjava) convReluPoolBackward.get("db");
		// Add regularization
		numjava regw1 = numjava.MulbyVal(w1, reg);
		numjava regw2 = numjava.MulbyVal(w2, reg);
		dw1 = numjava.add(dw1, regw1);
		dw2 = numjava.add(dw2, regw2);
		double w1Sum = numjava.sum(numjava.sum(numjava.elementmul(w1, w1), 0),1).finalmatrix[0][0];
		double w2Sum = numjava.sum(numjava.sum(numjava.elementmul(w2, w2), 0),1).finalmatrix[0][0];
		// To get only sum of all the elements
		double regLoss = (double)0.5*reg*(w1Sum + w2Sum);
		double loss = dataLoss + regLoss;
		Map<String,Object> ret = new HashMap<String,Object>();
		ret.put("loss", loss);
		ret.put("w1", dw1);
		ret.put("w2", dw2);
		ret.put("b1",db1);
		ret.put("b2", db2);

		return ret;
	}


	/**
	 * Initialize model parameters for training convolutional neural nets
	 * @param weightScale scaling factor for weights
	 * @param biasScale scaling factor of bias
	 * @param noOfClasses Total number of classes
	 * @param noFilters Total number  of filters
	 * @param filterSize size of the filter
	 * @param channels Number of channels of the image
	 * @param width Width of the image
	 * @param height Height of the image
	 * 
	 * @return map containing values of w1,w2,b1,b2.
	 */
	public Map<String,Object>layerConvonet_loadParameters(double weightScale, double biasScale, int noOfClasses, int noFilters, int filterSize, int channels, int width, int height)

	{

		Map<String,Object> model = new HashMap<String,Object>();
		numjava w1 =	numjava.createrandom(noFilters,channels*filterSize*filterSize);
		w1 = numjava.MulbyVal(w1,weightScale);
		numjava b1 = numjava.createrandom(1, noFilters);
		b1 = numjava.MulbyVal(b1,biasScale);
		numjava w2 = numjava.createrandom((noFilters*width*height)/4, noOfClasses);
		w2 = numjava.MulbyVal(w2,weightScale);
		numjava b2 = numjava.createrandom(1,noOfClasses);
		b2 = numjava.MulbyVal(b2, biasScale);

		model.put("w1", w1);
		model.put("b1",b1);
		model.put("w2",w2);
		model.put("b2",b2);

		return model;
	}




}

