import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Test cases to all the layers with small data set.
 * @author suhaspillai
 *
 */
public class Test {

	layers layer = new layers();

	public void test() throws Exception
	{

		int numInputs = 2;
		int inputShape = 120; //(4,5,6)
		int outputDim = 3;
		int inputSize = numInputs * inputShape;
		int weightSize = outputDim * inputShape;
		numjava testArray_X= numjava.reshape(numjava.linspace(-(double)0.1, (double)0.5, inputSize),2,120);
		//  array of 2*120

		numjava weightArray = numjava.reshape(numjava.linspace(-(double)0.2, (double)0.3, weightSize),inputShape,outputDim);
		numjava bArray = numjava.reshape(numjava.linspace(-(double)0.3, (double)0.1, outputDim),1,outputDim); 
		Map<String,Object> resultAffineForward = layer.affine_forward(testArray_X, weightArray, bArray);			
		System.out.println("****************Affine Forward****************: ");
		numjava.print((numjava)resultAffineForward.get("out"));

		/*********************ReluForward******************************/

		numjava testArrayRelu_X= numjava.reshape(numjava.linspace(-(double)0.5, (double)0.5, 12),3,4);
		Map<String,Object>relu_ret = layer.relu_forward(testArrayRelu_X);
		System.out.println("***************Relu Forward*****************");
		numjava.print((numjava)relu_ret.get("out"));

		/************************SoftMaxLoss**************************/
		int num_classes = 10;
		int num_inputs = 50;
		numjava xtestsoftmax = numjava.MulbyVal(numjava.createrandom(50, 10),(double)0.001);
		int [] y1 = new int [50];
		numjava.getrandom(10, 50, y1);
		numjava y = new numjava(50,1); 

		for (int i= 0; i<50;i++)
		{
			y.finalmatrix[i][0] = y1[i];
		}
		Map<String,Object> m = layer.softmax_loss(xtestsoftmax, y); 
		System.out.println("************SoftMaxLoss*************");
		System.out.println("Loss is :"+m.get("loss"));




		/*********************ConvolutionForward*****************************/


		numjava testConvo_X = numjava.linspace((double)-0.1,(double)0.5,96);
		testConvo_X = numjava.reshape(testConvo_X, 2, 48);
		//		System.out.println("Printing X matrix");
		//		numjava.print(testConvo_X);
		numjava testConvo_w = numjava.linspace((double)-0.2,(double)0.3,144);
		testConvo_w = numjava.reshape(testConvo_w, 3, 48);
		//		System.out.println("Printing fliter matrix");
		//		numjava.print(testConvo_w);
		numjava testConvo_b = numjava.linspace((double)-0.1,(double)0.2,3);
		testConvo_b.finalmatrix[0][2] = (double)0.2;
		System.out.println("Printing bias matrix");
		numjava.print(testConvo_b);

		Map<String,Integer> convParam = new HashMap<String,Integer>();
		convParam.put("stride", 2);
		convParam.put("pad",1);
		convParam.put("filterHeight",4);
		convParam.put("filterWidth", 4);
		convParam.put("channels", 3);
		convParam.put("imageWidth", 4);
		convParam.put("imageHeight", 4);

		System.out.println("Convo Matrix");

		Map<String,Object> mconv = layer.conv_forward_naive(testConvo_X, testConvo_w, testConvo_b, convParam);
		System.out.println("******************Convolution Forward*********************");
		numjava xp = (numjava)mconv.get("out");
		numjava.print(xp);

		/*****************************MaxPool*********************************/

		numjava	xpool = numjava.linspace((double)-0.3,(double)0.4, 96);
		xpool=xpool.reshape(xpool, 2, 48);	
		Map<String,Integer> pool= new HashMap<String,Integer>();
		pool.put("poolWidth", 2);
		pool.put("poolHeight", 2);
		pool.put("stride",2);
		//Map<String,Object> out = layer.max_pool_forward(xpool, pool, 3, 4, 4);
		//numjava.print((numjava)out.get("x"));
		int pool_height = pool.get("poolHeight");
		int pool_width = pool.get("poolWidth");
		int stridepool = pool.get("stride");
		int W = 4;
		int H = 4;
		int pool_img_height = ((H-pool_height)/stridepool) + 1;
		int pool_img_width = ((W-pool_width)/stridepool)+1;
		int org_img_width = W;
		int org_img_height = H;
		int channelspool = 3;
		boolean flag = true;
		int rowInc = 0;
		int orgImgPixel = 0;
		int countColumnStride = 0;
		int countRowStride = 0;
		int maxPoolImgIndex = 0;
		numjava maxPool = new numjava (xpool.M,pool_img_height*pool_img_width);
		int colCount = 0;

		//maxPool.finalmatrix[0][0]=layer.getMaxValPixel(xpool, 0,countColumnStride,countRowStride,  stride,  orgImgPixel,  colCount,  pool_img_width,  org_img_width);
		Map<String,Object> out1 =null;
		out1 = layer.max_pool_forward(xpool, pool, 3, 4, 4);
		System.out.println("*****************MaxPool Forward********************");
		numjava.print((numjava)out1.get("out"));

		/**********************************************Testing affineBackward************************************************/

		//			layers layer  = new layers();
		numjava testAffineBck_X = numjava.createrandom(10, 6);
		numjava testAffineBck_w = numjava.createrandom(6,5);
		numjava testAffineBck_b = numjava.createrandom(1, 5);
		numjava dout = numjava.createrandom(10,5);

		Map<String,Object> affineForward_out = layer.affine_forward(testAffineBck_X, testAffineBck_w, testAffineBck_b);
		Map<String,Object> affineBackward_out = layer.affine_backward(dout, affineForward_out);
		numjava dx = (numjava)affineBackward_out.get("dx");
		numjava dw = (numjava)affineBackward_out.get("dw");
		numjava db = (numjava)affineBackward_out.get("db");
		numjava dx_num = gradientCheck.evalNumericalGradient_affineLayer("x", testAffineBck_X, testAffineBck_w, testAffineBck_b, dout, (double)0.00001);	
		numjava dw_num = gradientCheck.evalNumericalGradient_affineLayer("w", testAffineBck_X, testAffineBck_w, testAffineBck_b, dout, (double)0.00001);
		numjava db_num = gradientCheck.evalNumericalGradient_affineLayer("b", testAffineBck_X, testAffineBck_w, testAffineBck_b, dout, (double)0.00001);
		System.out.println("Gradient Check");
		System.out.println("dx error:  "+numjava.relError(dx_num,dx));
		System.out.println("dw error: "+numjava.relError(dw_num,dw));
		System.out.println("db error: "+numjava.relError(db_num,db));

		/**********************Testing Relu Backwards**************************/

		numjava testRelu_X = numjava.createrandom(10, 10);
		numjava doutRelu = numjava.createrandom(10,10);
		numjava dx_num_Relu = gradientCheck.evalNumericalGradient_ReluLayer(testRelu_X, doutRelu, 0.00001);
		Map<String,Object> ret_relu = layer.relu_forward(testRelu_X);
		numjava dx_relu = (numjava)layer.relu_backward(doutRelu, ret_relu).get("dout_relu");
		System.out.println("Checking relu backward");
		System.out.println("dx error: "+numjava.relError(dx_num_Relu, dx_relu));

		/*********************Testing Convolution Backward*********************/ 

		Map<String,Integer> conv_param = new HashMap<String,Integer>(); 
		conv_param.put("stride", 1);
		conv_param.put("pad",1);
		conv_param.put("filterHeight",3);
		conv_param.put("filterWidth", 3);
		conv_param.put("channels", 3);
		conv_param.put("imageWidth", 5);
		conv_param.put("imageHeight", 5);
		double hval = 0.00001;

		numjava testConvoBck_X = numjava.createrandom(3, 75);
		numjava ConvoBack_dout = numjava.createrandom(3,50);   
		numjava testConvoBck_w = numjava.createrandom(2,27);
		numjava testConvoBck_b = numjava.createrandom(1,2);
		mconv.clear();
		mconv = layer.conv_forward_naive(testConvoBck_X, testConvoBck_w, testConvoBck_b, conv_param);
		numjava x  = (numjava)mconv.get("x");
		numjava w = (numjava) mconv.get("w");
		numjava b = (numjava)mconv.get("b");
		numjava.print((numjava)mconv.get("out"));


		Map<String,Object> convo_ret = layer.conv_backward_naive(ConvoBack_dout, mconv);
		numjava dx_conv_analytical = (numjava)convo_ret.get("dx");
		numjava dw_conv_analytical = (numjava)convo_ret.get("dw");
		numjava db_conv_analytical = (numjava)convo_ret.get("db");

		//			numjava.print((numjava)convo_ret.get("dx"));
		//			numjava.print((numjava)convo_ret.get("dw"));
		//			numjava.print((numjava)convo_ret.get("db"));

		numjava dx_Convo_num = (numjava)gradientCheck.evalNumericalGradient_conv_forward_naive("x", testConvoBck_X, testConvoBck_w, testConvoBck_b, conv_param, ConvoBack_dout, hval);
		numjava dw_Convo_num = (numjava)gradientCheck.evalNumericalGradient_conv_forward_naive("w", testConvoBck_X, testConvoBck_w, testConvoBck_b, conv_param, ConvoBack_dout, hval);
		numjava db_Convo_num = (numjava)gradientCheck.evalNumericalGradient_conv_forward_naive("b", testConvoBck_X, testConvoBck_w, testConvoBck_b, conv_param, ConvoBack_dout, hval);


		System.out.println("Relative error of Convolution");

		System.out.println("dx error: "+numjava.relError(dx_conv_analytical,dx_Convo_num ));
		System.out.println("dw error: "+numjava.relError(dw_conv_analytical,dw_Convo_num ));
		System.out.println("db error: "+numjava.relError(db_conv_analytical,db_Convo_num ));


		/***************************MaxPooling Backward***********************/
		pool.clear();
		pool= new HashMap<String,Integer>();
		pool.put("poolWidth", 2);
		pool.put("poolHeight", 2);
		pool.put("stride",2);
		W=4;
		H =4;
		int noOfChannels = 2;

		numjava maxPool_X  = numjava.createrandom(2,32);
		numjava maxPool_dout = numjava.createrandom(2,8);
		Map<String,Object> ret_out = layer.max_pool_forward(maxPool_X, pool, noOfChannels, W, H);
		numjava.print((numjava)ret_out.get("out"));
		Map<String,Object> maxpoolback=layer.max_pool_backward(maxPool_dout,ret_out, W, H, noOfChannels); 

		numjava dxpool = (numjava)maxpoolback.get("dx_maxpool");
		hval=0.00001;
		numjava dxpool_num = gradientCheck.evalNumericalGradient_Maxpool(maxPool_X, pool, noOfChannels, W, maxPool_dout, hval);
		System.out.println("Relative error max pool :"+numjava.relError(dxpool, dxpool_num));

		/************************SoftMax Error***************************/

		numjava testSoftMax_X = numjava.createrandom(1,10);

		numjava testSoftMax_y = new numjava (1,1);

		int	num_input = 1;
		hval = 0.00001;
		Random r = new Random();
		int rand;
		for (int i = 0; i<num_input ; i++)
		{
			rand = r.nextInt(10-0);
			//				System.out.println(rand+"   "+i);
			testSoftMax_y.finalmatrix[i][0]=rand;
		}
		numjava.print(testSoftMax_y);

		numjava testSoftMax_x_copy = numjava.deepCopy(testSoftMax_X);

		Map<String, Object> dxSoftmax = layer.softmax_loss(testSoftMax_x_copy,testSoftMax_y);
		numjava dx_SoftMax = (numjava)dxSoftmax.get("dx");
		//	testSoftMax_x_copy = numjava.deepCopy(testSoftMax_X);

		numjava dx_SoftMax_num = gradientCheck.evalNumericalGradient_SoftmaxLoss(testSoftMax_x_copy, testSoftMax_y, hval);
		System.out.println("Print analytical dx softmax: ");
		numjava.print(dx_SoftMax);
		System.out.println("Print numerical dx_num softmax: "+dx_SoftMax_num);
		numjava.print(dx_SoftMax_num);
		System.out.println("Testing SoftMax loss");
		System.out.println("dx_error: "+numjava.relError(dx_SoftMax_num, dx_SoftMax));


	}



}
