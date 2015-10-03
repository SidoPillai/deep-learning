import java.util.*;


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

}
