package core;

import java.util.Map;


public class gradientCheck {

	/**
	 * Evaluate numerical gradient for affine layer
	 * @param f : Output 
	 * @param x : X 
	 * @param df : derivative
	 */
	public static numjava evalNumericalGradient_affineLayer( String deriv_wrt, numjava x, numjava w, numjava b, numjava df,float hval) {
		
		numjava grad = new numjava(x.M,x.N);
		layers l = new layers();
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
						numjava pos  =  l.affine_forward(x, w, b).get("x"); // lambda function
						x.finalmatrix[i][j] = oldval - h;
						numjava neg = l.affine_forward(x, w, b).get("x"); // lambda function
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
			for (int i = 0; i<w.M; i++) 
			{
				for (int j = 0; j<w.N; j++)
				{
					oldval = w.finalmatrix[i][j];
					w.finalmatrix[i][j] = oldval + h;
					numjava pos  =  l.affine_forward(x, w, b).get("out"); // lambda function
					w.finalmatrix[i][j] = oldval - h;
					numjava neg = l.affine_forward(x, w, b).get("out"); // lambda function
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
			for (int i = 0; i<b.M; i++) 
			{
				for (int j = 0; j<b.N; j++)
				{
					oldval = b.finalmatrix[i][j];
					b.finalmatrix[i][j] = oldval + h;
					numjava pos  =  l.affine_forward(x, w, b).get("out"); // lambda function
					b.finalmatrix[i][j] = oldval - h;
					numjava neg = l.affine_forward(x, w, b).get("out"); // lambda function
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
	
	public static numjava evalNumericalGradient_ReluLayer( numjava x, numjava df,float hval)
	{
		
		numjava grad = new numjava(x.M,x.N);
		layers l = new layers();
		float oldval=0;
		float h = hval;
		for (int i = 0; i<x.M; i++) 
			{
				for (int j = 0; j<x.N; j++)
				{
					oldval = x.finalmatrix[i][j];
					x.finalmatrix[i][j] = oldval + h;
					numjava pos  =  l.relu_forward(x).get("out"); // lambda function
					x.finalmatrix[i][j] = oldval - h;
					numjava neg = l.relu_forward(x).get("out");// lambda function
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
	
	
	public static numjava evalNumericalGradient_conv_forward_naive(String deriv_wrt, numjava x, numjava w, numjava b, Map<String,Integer> conv_param, numjava df, float hval)
	{
		
		numjava grad = new numjava(x.M,x.N);
		layers l = new layers();
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
	
	
	public static numjava evalNumericalGradient_Maxpool( numjava x,Map<String,Integer> pool_param,int channels, int orgImgWidth, numjava df,float hval)
	{
		
		numjava grad = new numjava(x.M,x.N);
		layers l = new layers();
		float oldval=0;
		float h = hval;
		int orgImgHeight = orgImgWidth;
		for (int i = 0; i<x.M; i++) 
			{
				for (int j = 0; j<x.N; j++)
				{
					oldval = x.finalmatrix[i][j];
					x.finalmatrix[i][j] = oldval + h;
					numjava pos  =  (numjava)l.max_pool_forward(x, pool_param, channels, orgImgWidth, orgImgHeight).get("x"); // lambda function
					x.finalmatrix[i][j] = oldval - h;
					numjava neg = (numjava)l.max_pool_forward(x, pool_param, channels, orgImgWidth, orgImgHeight).get("x");// lambda function
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
}
	

	
