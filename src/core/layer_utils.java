import java.util.HashMap;
import java.util.Map;

/**
 * Class is used for creating different architectures of convolutional neural networks like (1 convolution
 * layers 1 relu 1 max pool or 1 convolution 1 relu 1 convolution 1 relu 1 max pool )
 */

public class layer_utils {


	public Map<String,Object>convReluForward(numjava x, numjava w, numjava b, Map<String,Integer> convParam)
	{
		layers layer = new layers();
		Map<String,Object> convRet = layer.conv_forward_naive(x, w, b, convParam);
		Map<String,Object> reluRet = layer.relu_forward((numjava)convRet.get("out"));
		Map<String,Object> cache = new HashMap<String,Object>();
		Map<String,Object> ret = new HashMap<String,Object>();
		cache.put("convX", convRet.get("x"));
		cache.put("convw", convRet.get("w"));
		cache.put("convb", convRet.get("b"));
		cache.put("conv_param", convRet.get("conv_param"));
		cache.put("reluCache", reluRet.get("cache"));
		ret.put("out", reluRet.get("out"));
		ret.put("cache", cache);

		return ret;

	}

	/**
	 * Forward propagation of the network
	 * @param x taring samples
	 * @param w weights
	 * @param b bias
	 * @param convParam values for convolutional layer
	 * @param poolParam values ofr pooling layer
	 * @param orgWidth Original width of the image
	 * @param orgHeight Original height of the image
	 * @return output values at convolution, relu and pooling layer.
	 */

	public Map<String,Object> convoReluPoolForward(numjava x, numjava w, numjava b, Map<String, Integer> convParam, Map<String, Integer> poolParam, int orgWidth, int orgHeight)
	{
		layers layer = new layers();
		Map<String,Object> convRet  =  new HashMap<String, Object>();
		Map<String,Object> reluRet = new HashMap<String, Object>();
		Map<String,Object> poolRet = new HashMap<String,Object>();
		Map<String,Object> cache = new HashMap<String,Object>();
		Map<String, Object> ret = new HashMap<String,Object>();


		convRet = layer.conv_forward_naive(x, w, b, convParam);
		//	Map<Integer,numjava> mconvout  = (Map<Integer,numjava>)convRet.get("out");
		reluRet = layer.relu_forward((numjava)convRet.get("out"));
		poolRet = layer.max_pool_forward((numjava)reluRet.get("out"), poolParam, w.M, orgWidth, orgHeight);
		convRet.remove("out");
		reluRet.remove("out");

		cache.put("convCache", convRet);
		cache.put("reluCache", reluRet);
		cache.put("poolCache", poolRet);
		//Return 
		ret.put("out",poolRet.get("out"));
		ret.put("cache", cache);

		return ret;

	}

	/**
	 * Backward propagation
	 * @return map containing derivatives of convolutional, relu and max pooling layer.
	 */

	public Map<String,Object>convoReluPoolBackward(numjava dout, Map<String,Object> cacheConvReluForward, int orgimgWidth, int orgimgHeight)
	{
		layers layer = new layers();
		Map<String,Object> cache = (Map<String,Object>)cacheConvReluForward.get("cache");
		Map<String,Object> convCache = (Map<String,Object>)cache.get("convCache");
		Map<String,Object> reluCache = (Map<String,Object>)cache.get("reluCache");
		Map<String,Object> poolCache = (Map<String,Object>)cache.get("poolCache");

		Map<String,Object> dout_max = layer.max_pool_backward(dout, poolCache, orgimgWidth, orgimgHeight, 3);

		Map<String,Object> dout_relu = layer.relu_backward((numjava)dout_max.get("dx_maxpool"),reluCache);

		Map<String,Object> conv_ret = layer.conv_backward_naive((numjava)dout_relu.get("dout_relu"), convCache);

		return conv_ret;


	}

	/*
	 * Forward propagation.
	 */
	public Map<String,Object> affineForward(numjava x, numjava w, numjava b)
	{
		layers layer = new layers();
		Map<String,Object> ret = layer.affine_forward(x, w, b);
		return ret;
	}




}
