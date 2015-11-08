package core;

import java.util.Map;


public class GradientCheck {

	/**
	 * Evaluate numerical gradient for affine layer
	 * @param f : Output 
	 * @param x : X 
	 * @param df : derivative
	 */
	public static NumJava evalNumericalGradient_affineLayer( String deriv_wrt, NumJava x, NumJava w, NumJava b, NumJava df,float hval) {
		
		NumJava grad = new NumJava(x.M,x.N);
		Layers l = new Layers();
		float oldval=0;
		float h = hval;
		
		if(deriv_wrt=="x")
		{
			for (int i = 0; i<x.M; i++) 
				{
					for (int j = 0; j<x.N; j++)
					{
						oldval = x.finalmatrix[i][j];
						x.finalmatrix[i][j] = oldval + h;
						NumJava pos  =  l.affine_forward(x, w, b).get("x"); // lambda function
						x.finalmatrix[i][j] = oldval - h;
						NumJava neg = l.affine_forward(x, w, b).get("x"); // lambda function
						x.finalmatrix[i][j] = oldval;
						NumJava mat = NumJava.elementMul(NumJava.sub(pos, neg), df);
						// first sum across rows then across columns to get just one value at 0,0 
						grad.finalmatrix[i][j] = NumJava.divideByVal(NumJava.sum(NumJava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
								// accumulate sum for entire matrix; 
					// This is being calculated above (pos-neg)*df/2*h;
						
					}
				}
		}
		else if (deriv_wrt=="w")
		{
			for (int i = 0; i<w.M; i++) 
			{
				for (int j = 0; j<w.N; j++)
				{
					oldval = w.finalmatrix[i][j];
					w.finalmatrix[i][j] = oldval + h;
					NumJava pos  =  l.affine_forward(x, w, b).get("out"); // lambda function
					w.finalmatrix[i][j] = oldval - h;
					NumJava neg = l.affine_forward(x, w, b).get("out"); // lambda function
					w.finalmatrix[i][j] = oldval;
					NumJava mat = NumJava.elementMul(NumJava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = NumJava.divideByVal(NumJava.sum(NumJava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
							// accumulate sum for entire matrix; 
				// This is being calculated above (pos-neg)*df/2*h;
					
				}
			}
		}
		else if (deriv_wrt=="b")
		{
			for (int i = 0; i<b.M; i++) 
			{
				for (int j = 0; j<b.N; j++)
				{
					oldval = b.finalmatrix[i][j];
					b.finalmatrix[i][j] = oldval + h;
					NumJava pos  =  l.affine_forward(x, w, b).get("out"); // lambda function
					b.finalmatrix[i][j] = oldval - h;
					NumJava neg = l.affine_forward(x, w, b).get("out"); // lambda function
					b.finalmatrix[i][j] = oldval;
					NumJava mat = NumJava.elementMul(NumJava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = NumJava.divideByVal(NumJava.sum(NumJava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
							// accumulate sum for entire matrix; 
				// This is being calculated above (pos-neg)*df/2*h;
					
				}
			}
		}
		l=null;
		return grad;
		
	}
	
	public static NumJava evalNumericalGradient_ReluLayer( NumJava x, NumJava df,float hval)
	{
		
		NumJava grad = new NumJava(x.M,x.N);
		Layers l = new Layers();
		float oldval=0;
		float h = hval;
		for (int i = 0; i<x.M; i++) 
			{
				for (int j = 0; j<x.N; j++)
				{
					oldval = x.finalmatrix[i][j];
					x.finalmatrix[i][j] = oldval + h;
					NumJava pos  =  l.relu_forward(x).get("out"); // lambda function
					x.finalmatrix[i][j] = oldval - h;
					NumJava neg = l.relu_forward(x).get("out");// lambda function
					x.finalmatrix[i][j] = oldval;
					NumJava mat = NumJava.elementMul(NumJava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = NumJava.divideByVal(NumJava.sum(NumJava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
							// accumulate sum for entire matrix; 
				// This is being calculated above (pos-neg)*df/2*h;
					
				}
			}
		l=null;
		return grad;
		
	}
	
	
	public static NumJava evalNumericalGradient_conv_forward_naive(String deriv_wrt, NumJava x, NumJava w, NumJava b, Map<String,Integer> conv_param, NumJava df, float hval)
	{
		
		NumJava grad = new NumJava(x.M,x.N);
		Layers l = new Layers();
		float oldval=0;
		float h = hval;
		
		if (deriv_wrt=="x")
		{
			for (int i = 0; i<x.M; i++) 
				{
					for (int j = 0; j<x.N; j++)
					{
						oldval = x.finalmatrix[i][j];
						x.finalmatrix[i][j] = oldval + h;
						NumJava pos  =  (NumJava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
						x.finalmatrix[i][j] = oldval - h;
						NumJava neg =  (NumJava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
						x.finalmatrix[i][j] = oldval;
						NumJava mat = NumJava.elementMul(NumJava.sub(pos, neg), df);
						// first sum across rows then across columns to get just one value at 0,0 
						grad.finalmatrix[i][j] = NumJava.divideByVal(NumJava.sum(NumJava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
								// accumulate sum for entire matrix; 
					// This is being calculated above (pos-neg)*df/2*h;
						
					}
				}
		}
		else if(deriv_wrt=="w")
		{
			for (int i = 0; i<w.M; i++) 
			{
				for (int j = 0; j<w.N; j++)
				{
					oldval = w.finalmatrix[i][j];
					w.finalmatrix[i][j] = oldval + h;
					NumJava pos  =  (NumJava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					w.finalmatrix[i][j] = oldval - h;
					NumJava neg =  (NumJava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					w.finalmatrix[i][j] = oldval;
					NumJava mat = NumJava.elementMul(NumJava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = NumJava.divideByVal(NumJava.sum(NumJava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
							// accumulate sum for entire matrix; 
				// This is being calculated above (pos-neg)*df/2*h;
					
				}
			}
		}
		else if(deriv_wrt=="b")
		{
			for (int i = 0; i<b.M; i++) 
			{
				for (int j = 0; j<b.N; j++)
				{
					oldval = b.finalmatrix[i][j];
					b.finalmatrix[i][j] = oldval + h;
					NumJava pos  =  (NumJava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					b.finalmatrix[i][j] = oldval - h;
					NumJava neg =  (NumJava)l.conv_forward_naive(x, w, b, conv_param).get("out");// lambda function
					b.finalmatrix[i][j] = oldval;
					NumJava mat = NumJava.elementMul(NumJava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = NumJava.divideByVal(NumJava.sum(NumJava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
							// accumulate sum for entire matrix; 
				// This is being calculated above (pos-neg)*df/2*h;
					
				}
			}
		}
			
		l=null;
		return grad;
		
	}
	
	
	public static NumJava evalNumericalGradient_Maxpool( NumJava x,Map<String,Integer> pool_param,int channels, int orgImgWidth, NumJava df,float hval)
	{
		
		NumJava grad = new NumJava(x.M,x.N);
		Layers l = new Layers();
		float oldval=0;
		float h = hval;
		int orgImgHeight = orgImgWidth;
		for (int i = 0; i<x.M; i++) 
			{
				for (int j = 0; j<x.N; j++)
				{
					oldval = x.finalmatrix[i][j];
					x.finalmatrix[i][j] = oldval + h;
					NumJava pos  =  (NumJava)l.max_pool_forward(x, pool_param, channels, orgImgWidth, orgImgHeight).get("x"); // lambda function
					x.finalmatrix[i][j] = oldval - h;
					NumJava neg = (NumJava)l.max_pool_forward(x, pool_param, channels, orgImgWidth, orgImgHeight).get("x");// lambda function
					x.finalmatrix[i][j] = oldval;
					NumJava mat = NumJava.elementMul(NumJava.sub(pos, neg), df);
					// first sum across rows then across columns to get just one value at 0,0 
					grad.finalmatrix[i][j] = NumJava.divideByVal(NumJava.sum(NumJava.sum(mat, 0),1),2*h).finalmatrix[0][0]; 
							// accumulate sum for entire matrix; 
				// This is being calculated above (pos-neg)*df/2*h;
					
				}
			}
		l=null;
		return grad;
		
	}
	
	//For softMax layer
	public static numjava evalNumericalGradient_SoftmaxLoss( numjava x, numjava y,float hval)
	{
		
		numjava grad = new numjava(x.M,x.N); // create a matrix
		layers l = new layers();
		float oldval=0;
		float h = hval;
		float loss_fx_plusH = 0; 	// loss value form soft max function.
		float loss_fx_minusH = 0;   // loss value from soft max
		for (int i = 0; i<x.M; i++) 
			{
				for (int j = 0; j<x.N; j++)
				{
					oldval = x.finalmatrix[i][j];
					x.finalmatrix[i][j] = oldval + h;
					loss_fx_plusH  =  (float)l.softmax_loss(x, y).get("loss"); 
					x.finalmatrix[i][j] = oldval - h;
					loss_fx_minusH  =  (float)l.softmax_loss(x, y).get("loss");
					x.finalmatrix[i][j] = oldval;
					
					// Change in loss when a value of x is incremented and decremented by a small value. 
					grad.finalmatrix[i][j] = (loss_fx_plusH-loss_fx_minusH)/(2*h) ;
												
				}
			}
		l=null;
		return grad;
		
	}
	
}
	

	
