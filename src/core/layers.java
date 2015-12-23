
import java.util.*;

/**
 * Layers class contains all the methods to calculate forward and backward propagation for all the layers 
 * in convolutional neural network.
 * @author suhaspillai
 *
 */

public class layers {

	Map<String,Object> cache;

	public layers()
	{
		cache = new HashMap<String,Object>();
	}

	/**
	 * Computes forward propagation for affine layer (fully connected layer)
	 * @param x Number of training examples
	 * @param w weights 
	 * @param b bias
	 * @return map containing values out, x,w, and b
	 */

	public Map <String,Object> affine_forward(numjava x,numjava w, numjava b)
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

	/**
	 * Computes backward pass for affine layer
	 * @param dout output values
	 * @param map containing values from forward pass
	 */

	public  Map<String,Object> affine_backward(numjava dout, Map<String, Object> cache)
	{
		numjava x = (numjava)cache.get("x");
		numjava w = (numjava)cache.get("w");
		numjava b = (numjava)cache.get("b");

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

		Map<String, Object> ret = new HashMap<String, Object>();
		ret.put("dx", dx);
		ret.put("dw", dw);
		ret.put("db", db);

		return ret;
	}

	/**
	 * Computes forward pass for rectified linear unit
	 * @param input training sample
	 * @return map containing output of same shape .
	 */


	public Map<String,Object> relu_forward(numjava x)
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
		Map<String,Object> ret = new HashMap<String, Object>();
		ret.put("out", out);
		ret.put("x", x);
		return ret;

	}

	/*
	 * Compute backward pass for relu function.
	 */

	public Map<String,Object> relu_backward(numjava dout, Map<String,Object> cache)
	{
		numjava dx = null;
		numjava x = (numjava)cache.get("x");
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

		Map<String, Object> ret = new HashMap<String, Object>();
		ret.put("dout_relu", doutret);
		return ret;

	}

	/**
	 * The method is an implementation of convolutional forward pass.
	 * @param x input image 
	 * @param w weights
	 * @param b bias
	 * @param conv_param map containing values of filters performing convolution.
	 * @return map with an image of depth equal to number of filters and width and height equal to that of an original image.
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
		int noOfFilters = w.M; // this needs to be corrected
		int odimWidth = 1 + (conv_param.get("imageWidth")-filterW + 2 * pad)/stride; // output dimension of the convolved image.
		int odimHeight = 1  + (conv_param.get("imageHeight")-filterH + 2 * pad)/stride;
		numjava im2col = new numjava(filterH*filterW*channels, odimWidth*odimHeight);	
		Map<Integer,numjava> outMapim2col = new HashMap<Integer, numjava>();
		numjava xnew;
		int val = 0; // setting pad value as zero
		int imHeight = conv_param.get("imageHeight"); // Considering 32 by 32 pixel image.
		numjava output = new numjava(Nrow,noOfFilters*odimWidth*odimHeight);  // This is like converting 96 by 3072 into (96*3072)
		int imWidth = conv_param.get("imageWidth"); // Considering 32 by 32 pixel image.
		for (int i = 0; i<Nrow ; i++)
		{
			numjava convomat;			
			xnew = numjava.pad(x.finalmatrix[i], pad, 0 , channels, imWidth, imHeight); // pad the image
			// make it im2col for dot matrix multiplication.
			im2col = numjava.im2col(im2col, xnew, odimWidth, odimHeight, channels, filterW, noOfFilters,0, stride, pad, imWidth, imHeight);
			numjava dot_prod = numjava.dot(w,im2col);
			convomat = numjava.addByRow(dot_prod,b);
			convomat = numjava.reshape(convomat, 1,noOfFilters*odimWidth*odimHeight);
			output.finalmatrix[i]= convomat.finalmatrix[0];  // this will be the out out (i.e no of fileter by dimension of output image)
		}

		Map<String, Object> ret = new HashMap<String, Object>();
		ret.put("x", x);
		ret.put("w", w);
		ret.put("b", b);
		ret.put("out",output);
		ret.put("conv_param",conv_param);

		return ret;

	}

	/**
	 * The method performs back ward pass of convolution layer.
	 * @param dout upstream derivatives 
	 * @param cache map containing values of x, w, and b
	 * @return gradients dx with respect to x , dw with respect to w and db with respect to b
	 */

	public Map<String, Object> conv_backward_naive(numjava dout_out, Map<String, Object> cache)
	{

		numjava x  = (numjava)cache.get("x");
		numjava w = (numjava) cache.get("w");
		numjava b = (numjava)cache.get("b");
		Map<String,Integer> conv_param = (Map<String,Integer>)cache.get("conv_param");
		numjava temp = new numjava(1,dout_out.N);
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
		numjava im2col = new numjava(filterH*filterW*channels, odimWidth*odimHeight);
		numjava xnew;
		int imWidth = conv_param.get("imageWidth"); // Considering 32 by 32 pixel image. // you have to change this
		int imHeight = conv_param.get("imageHeight"); // Considering 32 by 32 pixel image.
		numjava dw = new numjava(w.M,w.N);
		numjava dx = new numjava(x.M,x.N);
		numjava db = new numjava(b.M,b.N);
		numjava x_deconvolve = new numjava (im2col.M,im2col.N); // im2col dimension
		numjava originalMat = new numjava (x.M,x.N); // Get the image without any padding
		numjava dout=null;
		for (int i = 0; i<Nrow ; i++)
		{
			temp.finalmatrix[0] = dout_out.finalmatrix[i];
			dout = numjava.reshape(temp, w.M, odimWidth*odimHeight);
			xnew = numjava.pad(x.finalmatrix[i], pad, 0 , channels, imWidth, imHeight); // pad the image
			// make it im2col for dot matrix multiplication.
			im2col = numjava.im2col(im2col, xnew, odimWidth, odimHeight, channels, filterW, noOfFilters,0, stride, pad, imWidth, imHeight);
			dw = numjava.add(dw, numjava.dot(dout,numjava.transpose(im2col))); //check this
			db = numjava.add(db, numjava.reshape(numjava.sum(dout, 1),db.M,db.N));
			x_deconvolve = numjava.dot(numjava.transpose(w),dout);// check this
			// Actually, you can only call the method deConvolve and pass original mat as an array ref, which will then be filled and get that array 
			// which will be dx, no need to explicitly assign value to dx
			dx.finalmatrix[i] = numjava.deConvolve(originalMat.finalmatrix[i], x_deconvolve, pad, filterW, channels, stride, odimWidth, imHeight, imWidth);
		}
		Map<String, Object> ret = new HashMap<String, Object>();
		ret.put("dx", dx);
		ret.put("dw", dw);
		ret.put("db", db);
		return ret;
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
	public  double getMaxValPixel(numjava x, int imgNo, int countColumnStride,int countRowStride, int stride, int orgImgPixel, int colCount, int pool_img_width, int org_img_width)
	{
		double [] maxPoolStore =new double [pool_img_width*pool_img_width];
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
			//orgImgPixel = stride*colCount + (org_img_width-pool_img_width);
			orgImgPixel = orgImgPixel + (org_img_width-pool_img_width);
		}

		double max = maxPoolStore[0];
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

		int position=orgImgPixel;
		double maxVal;
		double prevMax = -1;
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
			}
			countColumnStride = 0;
			countRowStride++;
			// So suppose you have a 2*2 filter, we get the [(0,0),(0,1)] and img size is 4 * 4, value store in the array [(1,0),(1,1)] is obtained by skipping the other values
			//in that row and going to the 1st row to get [(1,0),(1,1)], so adding width below
			//orgImgPixel = stride*colCount + org_img_width;
			orgImgPixel = orgImgPixel + (org_img_width-pool_img_width);// corrected
		}
		return position;

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
		int pool_height = pool_param.get("poolHeight");
		int pool_width = pool_param.get("poolWidth");
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
		numjava maxPool = new numjava (x.M,noOfChannels*pool_img_height*pool_img_width); // Array to return after maxpool.
		int rowIncpoolimage=0;
		for(int img= 0; img<x.M; img++)  // For all images.
		{
			flag = true;
			for(int rowCount = 0; rowCount<pool_img_height; rowCount++)
			{
				if(flag == true)
				{
					rowInc = 0;
					rowIncpoolimage = 0;
					flag = false;
				}
				else
				{
					rowInc = rowInc + W * stride; // Basically, for row traversing
					rowIncpoolimage = rowIncpoolimage + pool_img_width; // for pooled image.
				}
				for (int colCount = 0; colCount < pool_img_width; colCount++)
				{
					orgImgPixel = colCount * stride + rowInc; // For column traversing of an image.
					maxPoolImgIndex = colCount + rowIncpoolimage;
					while (channels <= noOfChannels)
					{
						maxPool.finalmatrix[img][maxPoolImgIndex] = getMaxValPixel(x, img, countColumnStride,countRowStride, stride, orgImgPixel, colCount, pool_width, org_img_width);
						// The below statement helps in going from R(1024) -> G(1025 - 2048) -> B ( 2048 - 3072 ) i.e  
						//orgImgPixel = stride * colCount + (org_img_width * org_img_height)*channels;
						orgImgPixel = (colCount * stride + rowInc) + (org_img_width * org_img_height)*channels;
						// So here down scaling the image i.e if 32*32 - >16 * 16 img ---> 8 * 8 image, pool_img_width is used 
						// The below statement helps in going from R(1024) -> G(1025 - 2048) -> B ( 2048 - 3072 )
						//maxPoolImgIndex = colCount + (pool_img_width * pool_img_height)*channels;
						maxPoolImgIndex = (colCount + rowIncpoolimage) + (pool_img_width * pool_img_height)*channels;
						channels++;
					}
					channels = 1;
				}

			}

		}

		Map<String,Object> ret = new HashMap<String, Object>();
		ret.put("out", maxPool);
		ret.put("x", x);
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
	public  Map<String, Object> max_pool_backward (numjava dout, Map<String,Object> cache, int W, int H, int noOfChannels)
	{
		numjava x = (numjava)cache.get("x");
		numjava dx = new numjava(x.M,x.N); // output array
		Map<String,Integer> pool_param = (Map<String,Integer>)cache.get("pool_param");
		int pool_width = pool_param.get("poolWidth");
		int pool_height = pool_param.get("poolHeight");
		int stride = pool_param.get("stride");
		int pool_img_height = ((H-pool_height)/stride) + 1;
		int pool_img_width = ((W-pool_width)/stride)+1;
		int org_img_width = W;
		int org_img_height = W;
		int channels =1;
		boolean flag = true;
		int rowInc = 0;
		int orgImgPixel = 0;
		int countColumnStride = 0;
		int countRowStride = 0;
		int maxPoolImgIndex = 0;
		int position;
		int rowIncpoolimage=0;
		for(int img= 0; img<x.M; img++)  // For all images.
		{	
			flag = true;
			for(int rowCount = 0; rowCount<pool_img_height; rowCount++)
			{
				if(flag == true)
				{
					rowInc = 0;
					flag = false;
					rowIncpoolimage=0;
				}
				else
				{
					rowInc = rowInc + W * stride; // Basically, for row traversing , original image.
					rowIncpoolimage = rowIncpoolimage + pool_img_width;// for getting value from dout
				}
				for (int colCount = 0; colCount < pool_img_width; colCount++)
				{
					maxPoolImgIndex = colCount + rowIncpoolimage;
					orgImgPixel = colCount * stride + rowInc; // For column traversing of an image.
					while (channels <= noOfChannels) // for R,G,B channels
					{
						// Get the position of the max value of the pixels, which was obtained when max pooling was done on original image.
						position = getMaxValPixelPosition(x, img, countColumnStride,countRowStride, stride, orgImgPixel, colCount, pool_width, org_img_width);

						// put value at that position, which is there in the derivative matrix dout.
						dx.finalmatrix[img][position] = dout.finalmatrix[img][maxPoolImgIndex];
						//maxPoolImgIndex = colCount + (pool_img_width * pool_img_height)*channels;
						maxPoolImgIndex = (colCount + rowIncpoolimage) + (pool_img_width * pool_img_height)*channels;
						orgImgPixel = (colCount * stride + rowInc) + (org_img_width * org_img_height)*channels;
						channels++;
					}
					channels = 1;	
				}


			}		

		} // end of first for

		Map<String,Object> ret = new HashMap<String,Object>();
		ret.put("dx_maxpool", dx);
		return ret;
	}	



	/**
	 * The method computes softmax loss 
	 * @param x input to final layer
	 * @param y labels for the images
	 * @return map containing loss and derivatives at output layer (final layer).
	 */

	public  Map<String,Object> softmax_loss(numjava x, numjava y)
	{
		// Computes loss and gradients for softmax classification.

		numjava probs = new numjava(x.M,x.N);
		int M = x.M;
		double max;
		// subtracting max, so that we do not have large value for exponential, which would be numerically unstable. 
		for (int i = 0; i<M; i++)
		{
			max = numjava.get_Max_Element_Matrix_by_Column(x, i);
			probs.finalmatrix[i] = numjava.subByVal(x.finalmatrix[i],max);
		}
		probs = numjava.calculate_Exponential(probs);
		probs = numjava.divide(probs, numjava.sum(probs, 1));
		double loss = 0;
		int yVal;
		//Calculating total loss
		for (int i = 0; i<M; i++)
		{
			yVal = (int)y.finalmatrix[i][0];
			//yVal = (int)y.finalmatrix[0][i];
			loss = (double) (loss + (Math.log(probs.finalmatrix[i][yVal])));
		}
		loss = (double)-loss/M;

		numjava dx = numjava.deepCopy(probs); // derivative

		for (int i = 0; i<M;i++)
		{
			yVal = (int)y.finalmatrix[i][0];
			//yVal = (int)y.finalmatrix[0][i];
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