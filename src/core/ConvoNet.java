import java.util.HashMap;
import java.util.Map;


public class ConvoNet {

	
	public Map<String,Object> layerConvonet(numjava x, Map<String,Object> model, numjava y,float reg,int filterHeight, int orgimgWidth,int orgimgHeight,boolean check)
	{
		layer_utils layer_util = new layer_utils();
		layers layer = new layers();
		numjava w1 = (numjava)model.get("w1");
		numjava b1 = (numjava)model.get("b1");
		numjava w2 = (numjava)model.get("w2");
		numjava b2 = (numjava)model.get("b2");
		int M = x.M; // total no of images.
		int pad = (filterHeight-1)/2;
		Map<String,Integer> convParam = new HashMap<String,Integer>();
		Map<String,Integer> poolParam = new HashMap<String,Integer>();
//		Map<String, Object>    
		
		//convParam.put("stride", 1);
		convParam.put("pad", pad);
		poolParam.put("poolHeight",2);
		poolParam.put("poolWidth",2);
		poolParam.put("stride", 2);
	
		//Map<String,Object>convReluPoolCache = new HashMap<String,Object>();
		//Map<String,Object>affineForwardCache = new HashMap<String,Object>();
		
		
		//Defining original width
		//int orgWidth = 32;
		//Doing the convolution relu and pooling part.
		Map<String,Object>convReluPool = layer_util.convoReluPoolForward(x, w1, b1, convParam, poolParam, orgimgWidth, orgimgHeight);		
		
		//convReluPoolCache.put("cache1", convReluPool.get("cache"));
		
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

		
		//affineForwardCache.put("cache2",affineForward)
		
		// Now computing the soft max score
		// Here we get loss and the derivative of the soft max layer, which is used for back propagation. 
		Map<String,Object> softmaxout = layer.softmax_loss((numjava)affineForward.get("out"), y);
		float dataLoss =(float) softmaxout.get("loss");
		numjava dscores = (numjava)softmaxout.get("dx");
		
		
		//To compute the gradient using backward pass.
		//Sending affine forward to has out and cache obtained from affine forward. 
		Map<String,Object> affineBackward  = layer.affine_backward(dscores,affineForward);
		// Get derivative of w2,x, and b2 weight
		numjava dw2 = (numjava)affineBackward.get("dw");
		numjava dx2 = (numjava)affineBackward.get("dx");
		numjava db2  = (numjava)affineBackward.get("db");
		
		//int W =32;
		//int H = 32
		 
		Map<String,Object> convReluPoolBackward = new HashMap<String,Object>();
		//get derivative for conv mrelu and max pool
		numjava dw1 = (numjava)convReluPoolBackward.get("dw");
		numjava dx1 = (numjava) convReluPoolBackward.get("dx");
		numjava db1 = (numjava) convReluPoolBackward.get("db");
	
		// Add regularization
		// write method for multiplication by value.then do the last thing.
		numjava regw1 = numjava.MulbyVal(w1, reg);
		numjava regw2 = numjava.MulbyVal(w2, reg);
		dw1 = numjava.add(dw1, regw1);
		dw2 = numjava.add(dw2, regw2);
		float w1Sum = numjava.sum(numjava.sum(numjava.elementmul(w1, w1), 0),1).finalmatrix[0][0];
		float w2Sum = numjava.sum(numjava.sum(numjava.elementmul(w2, w2), 0),1).finalmatrix[0][0];
		// To get only sum of all the elements
		float regLoss = (float)0.5*reg*(w1Sum + w2Sum);
		
		float loss = dataLoss + regLoss;
		
		Map<String,Object> ret = new HashMap<String,Object>();
		ret.put("loss", loss);
		ret.put("w1", dw1);
		ret.put("w2", dw2);
		ret.put("b1",db1);
		ret.put("b2", db2);
		
		return ret;
	}
	
}
