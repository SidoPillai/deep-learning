package core;

import java.util.*;

/*
 * The class creates modular structure of all the layers of convolutionla neural nets
 *  @author suhaspillai
 *  @version 2-Oct-2015	
 */
public class Layers {

	Map<String,NumJava> cache;

	public Layers()
	{
		cache = new HashMap<String,NumJava>();
	}

	/*
	 * Computes forward pass for affine layer.
	 */
	public Map <String,NumJava> affine_forward(NumJava x,NumJava w, NumJava b)
	{
		NumJava out = new NumJava(x.M,w.N);
		out = NumJava.dot(x, w);
		out = NumJava.add(out, b);
		cache.put("out", out);
		cache.put("x", x);
		cache.put("w", w);
		cache.put("b", b);
		return cache; 
	}
	/*
	 * Computes backward pass for affine layer
	 */

	public  Map<String,NumJava> affine_backward(NumJava dout, Map<String, NumJava> cache)
	{
		NumJava x = cache.get("x");
		NumJava w = cache.get("w");
		NumJava b = cache.get("b");

		NumJava dx = null;
		NumJava dw = null;
		NumJava db = null;

		// Implement backward pass
		dx = new NumJava (x.M,x.N);
		dw = new NumJava (w.M,w.N);
		db = new NumJava (b.M,b.N);

		db = NumJava.add(db,NumJava.sum(dout, 0));
		dx = NumJava.dot(dout, NumJava.transpose(w));
		// every x1 x2 ...xn will be multiplied by dout1,dout2
		NumJava temp = NumJava.transpose(x);
		dw = NumJava.dot(temp,dout);

		Map<String, NumJava> ret = new HashMap<String,NumJava>();
		ret.put("dx", dx);
		ret.put("dw", dw);
		ret.put("db", db);

		return ret;
	}

	/*
	 * Computing forward pass for rectified linear unit
	 */

	public Map<String,NumJava> relu_forward(NumJava x)
	{
		NumJava out = null;
		out = new NumJava(x.M,x.N);
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
		Map<String,NumJava> ret = new HashMap<String, NumJava>();
		ret.put("out", out);

		return ret;

	}

	/*
	 * Compute backward pass for relu function.
	 */
	public Map<String,NumJava> relu_backward(NumJava dout, Map<String,NumJava> cache)
	{
		NumJava dx = null;
		NumJava x = cache.get("x");
		NumJava doutret = new NumJava(dout.M, dout.N);
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

		Map<String, NumJava> ret = new HashMap<String, NumJava>();
		ret.put("dout", doutret);
		return ret;

	}
	/**
	 * Compute convolution forward propagation. 
	*/
	public Map<String, Object> conv_forward_naive(NumJava x, NumJava w, NumJava b, Map<String, Integer> conv_param)
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
		NumJava im2col = new NumJava(noOfFilters,odimWidth * odimHeight);
		Map<Integer,NumJava> outMapim2col = new HashMap<Integer, NumJava>();
		NumJava xnew;
		int val = 0; // setting pad value as zero
		int imWidth = 32; // Considering 32 by 32 pixel image.
		int imHeight = 32; // Considering 32 by 32 pixel image.
		NumJava convomat=null;
		for (int i = 0; i<Nrow ; i++)
		{
			xnew = NumJava.pad(x.finalmatrix[i], pad, 0 , channels, imWidth, imHeight); // pad the image
			// make it im2col for dot matrix multiplication.
			im2col = NumJava.im2col(im2col, xnew, odimWidth, odimHeight, channels, filterW, noOfFilters,0, stride, pad, imWidth, imHeight);
			convomat = NumJava.add(NumJava.dot(w,im2col),b);
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
	 
	public Map<String, NumJava> conv_backward_naive(Map<Integer,NumJava> dout1, Map<String, Object> cache)
	{
	
		NumJava x  = (NumJava)cache.get("x");
		NumJava w = (NumJava) cache.get("w");
		NumJava b = (NumJava)cache.get("b");
		Map<String,Integer> conv_param = (Map<String,Integer>)cache.get("conv_param");
		Map<Integer,NumJava> dout = dout1;
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
		NumJava im2col = new NumJava(noOfFilters,odimWidth * odimHeight);
	//	Map<Integer,numjava> outMapim2col = new HashMap<Integer, numjava>();
		NumJava xnew;
		//int val = 0; // setting pad value as zero
		int imWidth = 32; // Considering 32 by 32 pixel image.
		int imHeight = 32; // Considering 32 by 32 pixel image.
		NumJava dw = new NumJava(w.M,w.N);
		NumJava dx = new NumJava(x.M,x.N);
		NumJava db = new NumJava(b.M,b.N);
		NumJava x_deconvolve = new NumJava (im2col.M,im2col.N); // im2col dimension
		NumJava originalMat = new NumJava (x.M,x.N); // Get the image without any padding
		
		for (int i = 0; i<Nrow ; i++)
		{
			xnew = NumJava.pad(x.finalmatrix[i], pad, 0 , channels, imWidth, imHeight); // pad the image
			// make it im2col for dot matrix multiplication.
			im2col = NumJava.im2col(im2col, xnew, odimWidth, odimHeight, channels, filterW, noOfFilters,0, stride, pad, imWidth, imHeight);
			dw = NumJava.add(dw, NumJava.dot(dout.get(i),NumJava.transpose(im2col)));
			db = NumJava.add(db, NumJava.sum(b, 1));
			x_deconvolve = NumJava.dot(NumJava.transpose(w),dout.get(i));
			dx = NumJava.deConvolve(originalMat, x_deconvolve, pad, filterW, channels, stride, odimWidth, imHeight, imWidth);
		}
		Map<String, NumJava> ret = new HashMap<String, NumJava>();
		ret.put("dx", dx);
		ret.put("dw", dw);
		ret.put("db", db);
		return ret;
	}
	
	public static float relError(NumJava a, NumJava b) throws Exception
	{
		NumJava relmat = new NumJava(a.M,a.N); 
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
	
	private static float get_Max_Element_Matrix_by_Value(NumJava relmat) {

		return 0;
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
	public  float getMaxValPixel(NumJava x, int imgNo, int countColumnStride,int countRowStride, int stride, int orgImgPixel, int colCount, int pool_img_width, int org_img_width)
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

	public  int getMaxValPixelPosition(NumJava x, int imgNo, int countColumnStride,int countRowStride, int stride, int orgImgPixel, int colCount, int pool_img_width, int org_img_width)
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
	public  Map<String,Object> max_pool_forward(NumJava x, Map<String,Integer> pool_param, int noOfChannels, int H, int W)
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
		NumJava maxPool = new NumJava (x.M,pool_img_height*pool_img_width); // Array to return after maxpool.

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
	public  Map<String, NumJava> max_pool_backward (NumJava dout, Map<String,Object> cache, int W, int H, int noOfChannels)
	{
		NumJava x = (NumJava)cache.get("x");
		NumJava dx = new NumJava(x.M,x.N); // output array
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

		Map<String,NumJava> ret = new HashMap<String,NumJava>();
		ret.put("dx", dx);
		return ret;
	}	


	/*
	 * Computing softMax loss
	 */

	public  Map<String,Object> softmax_loss(NumJava x, NumJava y)
	{
		// Computes loss and gradients for softsmax classification.

		NumJava probs = new NumJava(x.M,x.N);
		int M = x.M;
		float max;
		// subtracting max, so that we do not have large value for exponential, which would be numerically unstable. 
		for (int i = 0; i<M; i++)
		{
			max = NumJava.get_Max_Element_Matrix_by_Row(x, i);
			probs.finalmatrix[i] = NumJava.sub(x.finalmatrix[i],max);
		}
		probs = NumJava.calculate_Exponential(probs);
		probs = NumJava.divide(probs, NumJava.sum(probs, 1));
		float loss = 0;
		int yVal;
		//Calculating total loss
		for (int i = 0; i<M; i++)
		{
			yVal = (int)y.finalmatrix[i][0];
			loss = loss + probs.finalmatrix[i][yVal];
		}
		loss = -loss/M;

		NumJava dx = NumJava.deepCopy(probs); // derivative

		for (int i = 0; i<M;i++)
		{
			yVal = (int)y.finalmatrix[i][0];
			// For correct class derivative is (e(x)/sum(e(x)))-1 for rest it is e(x)/sum(e(x))
			dx.finalmatrix[i][yVal] = dx.finalmatrix[i][yVal] - 1;  

		}

		dx = NumJava.divideByVal(dx, M);

		Map<String,Object> ret = new HashMap<String,Object>();
		ret.put("loss", loss);
		ret.put("dx", dx);
		return ret;
	}

	
}	
