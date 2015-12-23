import java.util.Map;


/**
 * It is used for validating numerical and analytical gradient
 * @author suhaspillai
 *
 */
public class gradientCheck {

	/**
	 * 
	 * @param X Training data
	 * @param model model containing values for weigths and bias
	 * @param der_wrt derivative with respect to 
	 * @param hval small change in value
	 * @param y training label
	 * @param reg regularization parameter
	 * @param check if true, then get only the scores
	 * @param convParams map containing values only for convolution.
	 * @param verbose
	 * @return gradients with respect to that variable.
	 */

	public static numjava evalNumericalGradient(numjava X,Map<String,Object> model, String der_wrt,double hval,numjava y,double reg,boolean check,Map<String,Integer>convParam, boolean verbose)
	{
		ConvoNet conv = new ConvoNet();
		double oldval = 0;
		double h = hval;
		double loss1=0;
		double loss2=0;
		// get the grads for w1/w2/b1/b2
		numjava param = (numjava) model.get(der_wrt);
		numjava grad = new numjava(param.M,param.N);


		for(int i = 0; i<param.M; i++)
		{
			for(int j = 0; j < param.N; j++)
			{
				oldval = param.finalmatrix[i][j];
				param.finalmatrix[i][j] = oldval + h;
				loss1 = (double)conv.layerConvonet(X, model, y, reg, check, convParam).get("loss");
				param.finalmatrix[i][j] = oldval-h;
				loss2 = (double)conv.layerConvonet(X, model, y, reg, check, convParam).get("loss");
				param.finalmatrix[i][j] = oldval;
				grad.finalmatrix[i][j] = (loss1-loss2)/(2*h);

				if(verbose==true)
				{
					//	System.out.println("Value wrt:    "+der_wrt+"    "+grad.finalmatrix[i][j]);
				}

			}

		}
		return grad;
	}

	/**
	 * Evaluate numerical gradient for affine layer
	 * @param f : Output 
	 * @param x : X 
	 * @param df : derivative
	 */
	public static numjava evalNumericalGradient_affineLayer( String deriv_wrt, numjava x, numjava w, numjava b, numjava df,double hval)
	{


		layers l = new layers();
		double oldval=0;
		double h = hval;
		numjava grad=null;
		if(deriv_wrt=="x")
		{
			grad = new numjava(x.M,x.N);
			for (int i = 0; i<x.M; i++) 
			{
				for (int j = 0; j<x.N; j++)
				{
					oldval = x.finalmatrix[i][j];
					x.finalmatrix[i][j] = oldval + h;
					numjava pos  =  (numjava)l.affine_forward(x, w, b).get("out"); // lambda function
					x.finalmatrix[i][j] = oldval - h;
					numjava neg = (numjava)l.affine_forward(x, w, b).get("out"); // lambda function
					x.finalmatrix[i][j] = oldval;
					numjava mat = numjava.elementmul(numjava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = numjava.divideByVal(numjava.sum(numjava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
					// accumulate sum for entire matrix; 
					// This is being calculated above (pos-neg)*df/2*h;

				}
			}
		}
		else if (deriv_wrt=="w")
		{
			grad = new numjava(w.M,w.N);
			for (int i = 0; i<w.M; i++) 
			{
				for (int j = 0; j<w.N; j++)
				{
					oldval = w.finalmatrix[i][j];
					w.finalmatrix[i][j] = oldval + h;
					numjava pos  = (numjava) l.affine_forward(x, w, b).get("out"); // lambda function
					w.finalmatrix[i][j] = oldval - h;
					numjava neg = (numjava)l.affine_forward(x, w, b).get("out"); // lambda function
					w.finalmatrix[i][j] = oldval;
					numjava mat = numjava.elementmul(numjava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = numjava.divideByVal(numjava.sum(numjava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
					// accumulate sum for entire matrix; 
					// This is being calculated above (pos-neg)*df/2*h;

				}
			}
		}
		else if (deriv_wrt=="b")
		{
			grad = new numjava(b.M,b.N);
			for (int i = 0; i<b.M; i++) 
			{
				for (int j = 0; j<b.N; j++)
				{
					oldval = b.finalmatrix[i][j];
					b.finalmatrix[i][j] = oldval + h;
					numjava pos  =  (numjava)l.affine_forward(x, w, b).get("out"); // lambda function
					b.finalmatrix[i][j] = oldval - h;
					numjava neg = (numjava)l.affine_forward(x, w, b).get("out"); // lambda function
					b.finalmatrix[i][j] = oldval;
					numjava mat = numjava.elementmul(numjava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = numjava.divideByVal(numjava.sum(numjava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
					// accumulate sum for entire matrix; 
					// This is being calculated above (pos-neg)*df/2*h;

				}
			}
		}

		return grad;

	}

	/**
	 * Evaluate numerical gradient for the relu layer.
	 */

	public static numjava evalNumericalGradient_ReluLayer( numjava x, numjava df,double hval)
	{

		numjava grad = new numjava(x.M,x.N);
		layers l = new layers();
		double oldval=0;
		double h = hval;
		for (int i = 0; i<x.M; i++) 
		{
			for (int j = 0; j<x.N; j++)
			{
				oldval = x.finalmatrix[i][j];
				x.finalmatrix[i][j] = oldval + h;
				numjava pos  =  (numjava)l.relu_forward(x).get("out"); // lambda function
				x.finalmatrix[i][j] = oldval - h;
				numjava neg = (numjava)l.relu_forward(x).get("out");// lambda function
				x.finalmatrix[i][j] = oldval;
				numjava mat = numjava.elementmul(numjava.sub(pos, neg), df);
				// first sum across rows then across columns to get just one value at 0,0 
				grad.finalmatrix[i][j] = numjava.divideByVal(numjava.sum(numjava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
				// accumulate sum for entire matrix; 
				// This is being calculated above (pos-neg)*df/2*h;

			}
		}
		l=null;
		return grad;

	}

	/**
	 * Evaluate numerical gradient for convolutional layer.
	 */
	public static numjava evalNumericalGradient_conv_forward_naive(String deriv_wrt, numjava x, numjava w, numjava b, Map<String,Integer> conv_param, numjava df, double hval)
	{


		layers l = new layers();
		double oldval=0;
		double h = hval;
		numjava grad=null;
		if (deriv_wrt=="x")
		{
			grad = new numjava(x.M,x.N);
			for (int i = 0; i<x.M; i++) 
			{
				for (int j = 0; j<x.N; j++)
				{
					oldval = x.finalmatrix[i][j];
					x.finalmatrix[i][j] = oldval + h;
					numjava pos  =  (numjava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					x.finalmatrix[i][j] = oldval - h;
					numjava neg =  (numjava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					x.finalmatrix[i][j] = oldval;
					numjava mat = numjava.elementmul(numjava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = numjava.divideByVal(numjava.sum(numjava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
					// accumulate sum for entire matrix; 
					// This is being calculated above (pos-neg)*df/2*h;

				}
			}
		}
		else if(deriv_wrt=="w")
		{
			grad = new numjava(w.M,w.N);
			for (int i = 0; i<w.M; i++) 
			{
				for (int j = 0; j<w.N; j++)
				{
					oldval = w.finalmatrix[i][j];
					w.finalmatrix[i][j] = oldval + h;
					numjava pos  =  (numjava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					w.finalmatrix[i][j] = oldval - h;
					numjava neg =  (numjava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					w.finalmatrix[i][j] = oldval;
					numjava mat = numjava.elementmul(numjava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = numjava.divideByVal(numjava.sum(numjava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
					// accumulate sum for entire matrix; 
					// This is being calculated above (pos-neg)*df/2*h;

				}
			}
		}
		else if(deriv_wrt=="b")
		{
			grad = new numjava(b.M,b.N);
			for (int i = 0; i<b.M; i++) 
			{
				for (int j = 0; j<b.N; j++)
				{
					oldval = b.finalmatrix[i][j];
					b.finalmatrix[i][j] = oldval + h;
					numjava pos  =  (numjava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					b.finalmatrix[i][j] = oldval - h;
					numjava neg =  (numjava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					b.finalmatrix[i][j] = oldval;
					numjava mat = numjava.elementmul(numjava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = numjava.divideByVal(numjava.sum(numjava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
					// accumulate sum for entire matrix; 
					// This is being calculated above (pos-neg)*df/2*h;

				}
			}
		}

		l=null;
		return grad;

	}

	/**
	 * Evaluate numerical gradient for convolutional layer
	 */

	public static numjava evalNumericalGradient_Maxpool( numjava x,Map<String,Integer> pool_param,int channels, int orgImgWidth, numjava df,double hval)
	{

		numjava grad = new numjava(x.M,x.N);
		layers l = new layers();
		double oldval=0;
		double h = hval;
		int orgImgHeight = orgImgWidth;
		for (int i = 0; i<x.M; i++) 
		{
			for (int j = 0; j<x.N; j++)
			{
				oldval = x.finalmatrix[i][j];
				//	System.out.println("Before gradient plus h");
				//	numjava.print(x);
				x.finalmatrix[i][j] = oldval + h;
				numjava pos  =  (numjava)l.max_pool_forward(x, pool_param, channels, orgImgWidth, orgImgHeight).get("out"); // lambda function
				//	System.out.println("After plus h");
				//	numjava.print(x);

				x.finalmatrix[i][j] = oldval - h;
				numjava neg = (numjava)l.max_pool_forward(x, pool_param, channels, orgImgWidth, orgImgHeight).get("out");// lambda function
				//	System.out.println("After minus h");
				//	numjava.print(x);
				//	System.out.println("Value of pos :");
				//	numjava.print(pos);
				//	System.out.println("Value of neg: ");
				//	numjava.print(neg);

				x.finalmatrix[i][j] = oldval;
				numjava mat = numjava.elementmul(numjava.sub(pos, neg), df);
				// first sum across rows then across columns to get just one value at 0,0 

				grad.finalmatrix[i][j] = numjava.divideByVal(numjava.sum(numjava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
				//	System.out.println("Value at "+i+"   "+j+"is: "+grad.finalmatrix[i][j]);
				// accumulate sum for entire matrix; 
				// This is being calculated above (pos-neg)*df/2*h;

			}
		}
		l=null;
		return grad;

	}




	/**
	 * Evaluate numerical gradient for softmax layer.
	 */

	public static numjava evalNumericalGradient_SoftmaxLoss( numjava x, numjava y,double hval)
	{

		numjava grad = new numjava(x.M,x.N); // create a matrix
		layers l = new layers();
		double oldval=0;
		double h = hval;
		double loss_fx_plusH = 0; 	// loss value form soft max function.
		double loss_fx_minusH = 0;   // loss value from soft max
		numjava xCopy=null;
		System.out.println("Original x");
		numjava.print(x);

		for (int i = 0; i<x.M; i++) 
		{
			for (int j = 0; j<x.N; j++)
			{

				oldval = x.finalmatrix[i][j];
				xCopy = numjava.deepCopy(x);
				xCopy.finalmatrix[i][j] = oldval + h;

				loss_fx_plusH  =  (double)l.softmax_loss(xCopy, y).get("loss"); 

				xCopy = numjava.deepCopy(x);
				xCopy.finalmatrix[i][j] = oldval - h;
				loss_fx_minusH  =  (double)l.softmax_loss(xCopy, y).get("loss");
				xCopy=null;

				// Change in loss when a value of x is incremented and decremented by a small value. 
				grad.finalmatrix[i][j] = (loss_fx_plusH-loss_fx_minusH)/(2*h) ;

			}
		}
		l=null;
		return grad;

	}





}
