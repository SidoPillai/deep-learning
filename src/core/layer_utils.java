import java.util.HashMap;
import java.util.Map;


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
	
	
	public Map<String,Object> convoReluPoolForward(numjava x, numjava w, numjava b, Map<String, Integer> convParam, Map<String, Integer> poolParam, int orgWidth, int orgHeight)
	{
		layers layer = new layers();
		Map<String,Object> convRet  =  new HashMap<String, Object>();
		Map<String,Object> reluRet = new HashMap<String, Object>();
		Map<String,Object> poolRet = new HashMap<String,Object>();
		Map<String,Object> cache = new HashMap<String,Object>();
		Map<String, Object> ret = new HashMap<String,Object>();
		
		
		convRet = layer.conv_forward_naive(x, w, b, convParam);
		reluRet = layer.relu_forward((numjava)convRet.get("out"));
		poolRet = layer.max_pool_forward((numjava)reluRet.get("out"), poolParam, 3, orgWidth, orgHeight);
		convRet.remove("out");
		reluRet.remove("out");
		
		cache.put("convCache", convRet);
		cache.put("reluCache", reluRet);
		cache.put("poolCache", poolRet);
		//Return 
		ret.put("out",poolRet.get("x"));
		ret.put("cache", cache);
		
		return ret;
		
	}
	
	
	public Map<String,Object>convoReluPoolBackward(numjava dout, Map<String,Object> cache, int orgimgWidth, int orgimgHeight)
	{
		layers layer = new layers();
		Map<String,Object> convCache = (Map<String,Object>)cache.get("convCache");
		Map<String,Object> reluCache = (Map<String,Object>)cache.get("relucache");
		Map<String,Object> poolCache = (Map<String,Object>)cache.get("poolCache");
		
		numjava dout_max = (numjava)layer.max_pool_backward(dout, poolCache, orgimgWidth, orgimgHeight, 3);
		
		numjava dout_relu = (numjava)layer.relu_backward(dout_max,reluCache);
		
		Map<String,Object> conv_ret = layer.conv_backward_naive(dout_relu, convCache);
		
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
