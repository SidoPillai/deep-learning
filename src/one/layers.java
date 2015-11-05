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
	
	/**
	 * Computes backward propagation
	 * Date 10-27-2015
	 */
	 
	public Map<String, numjava> conv_backward_naive(Map<Integer,numjava> dout1, Map<String, Object> cache)
	{
	
		numjava x  = (numjava)cache.get("x");
		numjava w = (numjava) cache.get("w");
		numjava b = (numjava)cache.get("b");
		Map<String,Integer> conv_param = (Map<String,Integer>)cache.get("conv_param");
		Map<Integer,numjava> dout = dout1;
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
	//	Map<Integer,numjava> outMapim2col = new HashMap<Integer, numjava>();
		numjava xnew;
		//int val = 0; // setting pad value as zero
		int imWidth = 32; // Considering 32 by 32 pixel image.
		int imHeight = 32; // Considering 32 by 32 pixel image.
		numjava dw = new numjava(w.M,w.N);
		numjava dx = new numjava(x.M,x.N);
		numjava db = new numjava(b.M,b.N);
		numjava x_deconvolve = new numjava (im2col.M,im2col.N); // im2col dimension
		numjava originalMat = new numjava (x.M,x.N); // Get the image without any padding
		
		for (int i = 0; i<Nrow ; i++)
		{
			xnew = numjava.pad(x.finalmatrix[i], pad, 0 , channels, imWidth, imHeight); // pad the image
			// make it im2col for dot matrix multiplication.
			im2col = numjava.im2col(im2col, xnew, odimWidth, odimHeight, channels, filterW, noOfFilters,0, stride, pad, imWidth, imHeight);
			dw = numjava.add(dw, numjava.dot(dout.get(i),numjava.transpose(im2col)));
			db = numjava.add(db, numjava.sum(b, 1));
			x_deconvolve = numjava.dot(numjava.transpose(w),dout.get(i));
			dx = numjava.deConvolve(originalMat, x_deconvolve, pad, filterW, channels, stride, odimWidth, imHeight, imWidth);
		}
		Map<String, numjava> ret = new HashMap<String, numjava>();
		ret.put("dx", dx);
		ret.put("dw", dw);
		ret.put("db", db);
		return ret;
	}
	
	public static float relError(numjava a, numjava b) throws Exception
	{
		numjava relmat = new numjava(a.M,a.N); 
		for(int i = 0;i<a.M;i++)
		{
			for (int j = 0; j< a.N; j++)
			{
				
						float deno = (float)Math.max(0.00000001,Math.abs(a.finalmatrix[i][j])+Math.abs(b.finalmatrix[i][j]));
						relmat.finalmatrix[i][j] = Math.abs(a.finalmatrix[i][j]-b.finalmatrix[i][j])/deno;
			}
		}
		float max  = get_Max_Element_Matrix_by_Value(relmat);
		relmat = null;
		return max;
		
	}
	
	/**
	 * Get max pixel value when doing max pooling
	 * @param x Image matrix where every row is a pixel of 3072 if the image size is 32 * 32 
	 * @param imgNo Number of the image 
	 * @param countColumnStride Width of the max pool filter
	 * @param countRowStride Height of max pool filter
	 * @param stride Stride to take
	 * @param orgImgPixel pixel value position i.e any value between 3072 (if the image is 32 * 32)
	 * @param colCount every stride which we take while doing max pooling
	 * @param pool_img_width whidth of the image after max pooling
	 * @return Max value of the pixel 
	 */
	public  float getMaxValPixel(numjava x, int imgNo, int countColumnStride,int countRowStride, int stride, int orgImgPixel, int colCount, int pool_img_width, int org_img_width)
	{
		float [] maxPoolStore =new float [pool_img_width*pool_img_width];
		int index = 0;
		while (countRowStride < pool_img_width)
		{
			while (countColumnStride < pool_img_width)
			{
				maxPoolStore[index] =  x.finalmatrix[imgNo][orgImgPixel];
				countColumnStride++;
				orgImgPixel++;
				index++;
			}
			countColumnStride = 0;
			countRowStride++;
			// So suppose you have a 2*2 filter, we get the [(0,0),(0,1)] and img size is 4 * 4, value store in the array [(1,0),(1,1)] is obtained by skipping the other values
			//in that row and going to the 1st row to get [(1,0),(1,1)], so adding width below
			orgImgPixel = stride*colCount + org_img_width; 
		}

		float max = maxPoolStore[0];
		// Get the max value of [(0,0),(0,1),(1,0),(1,1)] if it is 2 by 2 pixel.
		for (int i = 1; i < maxPoolStore.length; i++)
		{
			if (max < maxPoolStore[i])
			{
				max = maxPoolStore[i];
			}
		}
		maxPoolStore = null;
		return max;

	}

	/**
	 * Get the max pixel position, when doing max pool.
	 * @param x
	 * @param x Image matrix where every row is a pixel of 3072 if the image size is 32 * 32 
	 * @param imgNo Number of the image 
	 * @param countColumnStride Width of the max pool filter
	 * @param countRowStride Height of max pool filter
	 * @param stride Stride to take
	 * @param orgImgPixel pixel value position i.e any value between 3072 (if the image is 32 * 32)
	 * @param colCount every stride which we take while doing max pooling
	 * @param pool_img_width whidth of the image after max pooling
	 * @return Position of the pixel that has max value. 
	 */

	public  int getMaxValPixelPosition(numjava x, int imgNo, int countColumnStride,int countRowStride, int stride, int orgImgPixel, int colCount, int pool_img_width, int org_img_width)
	{

		int index = 0;
		int position=0;
		float maxVal;
		float prevMax = -1;
		while (countRowStride < pool_img_width)
		{
			while (countColumnStride < pool_img_width)
			{
				maxVal = x.finalmatrix[imgNo][orgImgPixel];
				// Check which pixel has the max value and assign position of that element and return.
				if (prevMax<maxVal)
				{
					prevMax = maxVal;
					position = orgImgPixel;
				}
				countColumnStride++;
				orgImgPixel++;
				index++;
			}
			countColumnStride = 0;
			countRowStride++;
			// So suppose you have a 2*2 filter, we get the [(0,0),(0,1)] and img size is 4 * 4, value store in the array [(1,0),(1,1)] is obtained by skipping the other values
			//in that row and going to the 1st row to get [(1,0),(1,1)], so adding width below
			orgImgPixel = stride*colCount + org_img_width; 
		}
		return orgImgPixel;

	}

	/**
	 * Max pool forward
	 * @param x Image Matrix
	 * @param pool_param pooling parameters Map
	 * @param noOfChannels Number of channels in the image
	 * @param H Height of the original image before doing max pooling 
	 * @param W Width of the image before doing max pooling
	 * @return Map consisting of an image after pooling
	 */
	public  Map<String,Object> max_pool_forward(numjava x, Map<String,Integer> pool_param, int noOfChannels, int H, int W)
	{
		int pool_height = pool_param.get("pool_height");
		int pool_width = pool_param.get("pool_width");
		int stride = pool_param.get("stride");
		int pool_img_height = ((H-pool_height)/stride) + 1;
		int pool_img_width = ((W-pool_width)/stride)+1;
		int org_img_width = W;
		int org_img_height = H;
		int channels =1;
		boolean flag = true;
		int rowInc = 0;
		int orgImgPixel = 0;
		int countColumnStride = 0;
		int countRowStride = 0;
		int maxPoolImgIndex = 0;
		numjava maxPool = new numjava (x.M,pool_img_height*pool_img_width); // Array to return after maxpool.

		for(int img= 0; img<x.M; img++)  // For all images.
		{
			for(int rowCount = 0; rowCount<pool_height; rowCount++)
			{
				if(flag == true)
				{
					rowInc = 0;
					flag = false;
				}
				else
				{
					rowInc = rowInc + W * stride; // Basically, for row traversing 
				}
				for (int colCount = 0; colCount < pool_width; colCount++)
				{
					orgImgPixel = colCount * stride + rowInc; // For column traversing of an image.
					while (channels <= noOfChannels)
					{
						maxPool.finalmatrix[img][maxPoolImgIndex] = getMaxValPixel(x, rowCount, countColumnStride,countRowStride, stride, orgImgPixel, colCount, pool_img_width, org_img_width);
						// The below statement helps in going from R(1024) -> G(1025 - 2048) -> B ( 2048 - 3072 ) i.e  
						orgImgPixel = stride * colCount + (org_img_width * org_img_height)*channels;
						// So here down scaling the image i.e if 32*32 - >16 * 16 img ---> 8 * 8 image, pool_img_width is used 
						// The below statement helps in going from R(1024) -> G(1025 - 2048) -> B ( 2048 - 3072 )
						maxPoolImgIndex = colCount + (pool_img_width * pool_img_height)*channels;
						channels++;
					}
					channels = 1;
				}
			}
		}

		Map<String,Object> ret = new HashMap<String, Object>();
		ret.put("x", maxPool);
		ret.put("pool_param",pool_param);
		return ret;
	}

	/**
	 * Max pool backward
	 * @param dout derivative of the pooling layer
	 * @param cache 
	 * @param W W Width of the image before doing max pooling
	 * @param H Height of the original image before doing max pooling 
	 * @param noOfChannels Number of channels in the image
	 * @return Map containing dx values.
	 */
	public  Map<String, numjava> max_pool_backward (numjava dout, Map<String,Object> cache, int W, int H, int noOfChannels)
	{
		numjava x = (numjava)cache.get("x");
		numjava dx = new numjava(x.M,x.N); // output array
		Map<String,Integer> pool_param = (Map<String,Integer>)cache.get("pool_param");
		int pool_width = pool_param.get("pool_width");
		int pool_height = pool_param.get("pool_height");
		int stride = pool_param.get("stride");
		int pool_img_height = ((H-pool_height)/stride) + 1;
		int pool_img_width = ((W-pool_width)/stride)+1;
		int org_img_width = W;
		int channels =1;
		boolean flag = true;
		int rowInc = 0;
		int orgImgPixel = 0;
		int countColumnStride = 0;
		int countRowStride = 0;
		int maxPoolImgIndex = 0;
		int position;

		for(int img= 0; img<x.M; img++)  // For all images.
		{
			for(int rowCount = 0; rowCount<pool_height; rowCount++)
			{
				if(flag == true)
				{
					rowInc = 0;
					flag = false;
				}
				else
				{
					rowInc = rowInc + W * stride; // Basically, for row traversing 
				}
				for (int colCount = 0; colCount < pool_width; colCount++)
				{
					orgImgPixel = colCount * stride + rowInc; // For column traversing of an image.
					while (channels <= noOfChannels) // for R,G,B channels
					{
						// Get the position of the max value of the pixels, which was obtained when max pooling was done on original image.
						position = getMaxValPixelPosition(x, rowCount, countColumnStride,countRowStride, stride, orgImgPixel, colCount, pool_img_width, org_img_width);

						// put value at that position, which is there in the derivative matrix dout.
						dx.finalmatrix[img][position] = dout.finalmatrix[img][maxPoolImgIndex];
						maxPoolImgIndex = colCount + (pool_img_width * pool_img_height)*channels;
					}
					channels++;
				}
				channels = 1;

			}		

		} // end of first for

		Map<String,numjava> ret = new HashMap<String,numjava>();
		ret.put("dx", dx);
		return ret;
	}	


	/*
	 * Computing softMax loss
	 */

	public  Map<String,Object> softmax_loss(numjava x, numjava y)
	{
		// Computes loss and gradients for softsmax classification.

		numjava probs = new numjava(x.M,x.N);
		int M = x.M;
		float max;
		// subtracting max, so that we do not have large value for exponential, which would be numerically unstable. 
		for (int i = 0; i<M; i++)
		{
			max = numjava.get_Max_Element_Matrix_by_Row(x, i);
			probs.finalmatrix[i] = numjava.sub(x.finalmatrix[i],max);
		}
		probs = numjava.calculate_Exponential(probs);
		probs = numjava.divide(probs, numjava.sum(probs, 1));
		float loss = 0;
		int yVal;
		//Calculating total loss
		for (int i = 0; i<M; i++)
		{
			yVal = (int)y.finalmatrix[i][0];
			loss = loss + probs.finalmatrix[i][yVal];
		}
		loss = -loss/M;

		numjava dx = numjava.deepCopy(probs); // derivative

		for (int i = 0; i<M;i++)
		{
			yVal = (int)y.finalmatrix[i][0];
			// For correct class derivative is (e(x)/sum(e(x)))-1 for rest it is e(x)/sum(e(x))
			dx.finalmatrix[i][yVal] = dx.finalmatrix[i][yVal] - 1;  

		}

		dx = numjava.divideByVal(dx, M);

		Map<String,Object> ret = new HashMap<String,Object>();
		ret.put("loss", loss);
		ret.put("dx", dx);
		return ret;
	}

	
}	
	

	

}
