import java.util.*;

/*
 * The class creates modular structure of all the layers of convolutionla neural nets
 *  @author suhaspillai
 *  @version 2-Oct-2015	
 */
public class layers {

	Map<String,numjava> cache;

	public layers()
	{
		cache = new HashMap<String,numjava>();
	}

	/*
	 * Computes forward pass for affine layer.
	 */
	public Map <String,numjava> affine_forward(numjava x,numjava w, numjava b)
	{
		numjava out = new numjava(x.M,w.N);
		out = numjava.dot(x, w);
		out = numjava.add(out, b);
		cache.put("out", out);
		cache.put("x", x);
		cache.put("w", w);
		cache.put("b", b);
		return cache; 
	}
	/*
	 * Computes backward pass for affine layer
	 */

	public  Map<String,numjava> affine_backward(numjava dout, Map<String, numjava> cache)
	{
		numjava x = cache.get("x");
		numjava w = cache.get("w");
		numjava b = cache.get("b");

		numjava dx = null;
		numjava dw = null;
		numjava db = null;

		// Implement backward pass
		dx = new numjava (x.M,x.N);
		dw = new numjava (w.M,w.N);
		db = new numjava (b.M,b.N);

		db = numjava.add(db,numjava.sum(dout, 0));
		dx = numjava.dot(dout, numjava.transpose(w));
		// every x1 x2 ...xn will be multiplied by dout1,dout2
		numjava temp = numjava.transpose(x);
		dw = numjava.dot(temp,dout);

		Map<String, numjava> ret = new HashMap<String,numjava>();
		ret.put("dx", dx);
		ret.put("dw", dw);
		ret.put("db", db);

		return ret;
	}

	/*
	 * Computing forward pass for rectified linear unit
	 */

	public Map<String,numjava> relu_forward(numjava x)
	{
		numjava out = null;
		out = new numjava(x.M,x.N);
		for (int i = 0; i<x.M;i++)
		{
			for (int j = 0 ; j<x.N; j++)
			{
				if (x.finalmatrix[i][j]>0)
				{
					out.finalmatrix[i][j] = x.finalmatrix[i][j];
				}
			}

		}
		Map<String,numjava> ret = new HashMap<String, numjava>();
		ret.put("out", out);

		return ret;

	}

	/*
	 * Compute backward pass for relu function.
	 */
	public Map<String,numjava> relu_backward(numjava dout, Map<String,numjava> cache)
	{
		numjava dx = null;
		numjava x = cache.get("x");
		numjava doutret = new numjava(dout.M, dout.N);
		for (int i = 0; i<dout.M; i++)
		{
			for (int j=0; j<dout.N; j++)
			{
				if (x.finalmatrix[i][j]>0)
				{
					doutret.finalmatrix[i][j] = dout.finalmatrix[i][j];
				}
			}
		}

		Map<String, numjava> ret = new HashMap<String, numjava>();
		ret.put("dout", doutret);
		return ret;

	}
	/**
	 * Compute convolution forward propagation. 
	*/
	public Map<String, Object> conv_forward_naive(numjava x, numjava w, numjava b, Map<String, Integer> conv_param)
	{
		int stride = conv_param.get("stride");
		int pad = conv_param.get("pad");
		int filterH = conv_param.get("filterHeight");
		int filterW = conv_param.get("filterWidth");
		int channels = conv_param.get("channels");
		int Nrow = x.M;
		int Ncol = x.N;
		int noOfFilters = w.M;
		int odimWidth = 1 + (conv_param.get("imageWidth")-filterW + 2 * pad)/stride; // output dimension of the convolved image.
		int odimHeight = 1  + (conv_param.get("imageHeight")-filterH + 2 * pad)/stride;
		numjava im2col = new numjava(noOfFilters,odimWidth * odimHeight);
		Map<Integer,numjava> outMapim2col = new HashMap<Integer, numjava>();
		numjava xnew;
		int val = 0; // setting pad value as zero
		int imWidth = 32; // Considering 32 by 32 pixel image.
		int imHeight = 32; // Considering 32 by 32 pixel image.
		numjava convomat=null;
		for (int i = 0; i<Nrow ; i++)
		{
			xnew = numjava.pad(x.finalmatrix[i], pad, 0 , channels, imWidth, imHeight); // pad the image
			// make it im2col for dot matrix multiplication.
			im2col = numjava.im2col(im2col, xnew, odimWidth, odimHeight, channels, filterW, noOfFilters,0, stride, pad, imWidth, imHeight);
			convomat = numjava.add(numjava.dot(w,im2col),b);
			// store it in a map
			outMapim2col.put(i, convomat);
		}
		
		Map<String, Object> ret = new HashMap<String, Object>();
		ret.put("x", x);
		ret.put("w", w);
		ret.put("b", b);
		ret.put("out", outMapim2col);
		ret.put("conv_param",conv_param);
		
		return ret;
		
	}
	

}
