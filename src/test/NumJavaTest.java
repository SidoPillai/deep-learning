package test;

import core.NumJava;

public class NumJavaTest {

	public static void main(String[] args) {
		System.out.println("Creating a random matrix");
		NumJava obj = NumJava.createrandom(4, 4);
		NumJava.print(obj);
		System.out.println();
		System.out.println("Dot Product");
		NumJava matrix1 = NumJava.createrandom(3, 3);
		NumJava matrix2 = NumJava.createrandom(3, 3);
		System.out.println("Matrix 1");
		NumJava.print(matrix1);
		System.out.println("Matrix 2");
		NumJava.print(matrix2);
		System.out.println("Result");
		NumJava.print(NumJava.dot(matrix1, matrix2));
		System.out.println("Multiply Matrix 1 and 2");
		NumJava.print(NumJava.elementMul(matrix1, matrix2));
		System.out.println("Create Matrix");
		NumJava.creatematrix(matrix1.finalmatrix).printMe();
		NumJava.transpose(matrix1).printMe();
		System.out.println("Add");
		NumJava.add(matrix1, matrix2).printMe();
		System.out.println("Print Matrix 1");
		NumJava.print(matrix1);
		System.out.println("Sum Axis");
		NumJava.sum(matrix1, 0).printMe();
		NumJava.sum(matrix1, 1).printMe();
		System.out.println("Deep copy");
		NumJava matrix3 = matrix1.deepCopy(matrix2);
		matrix3.printMe();
		System.out.println("sub");
		float[] x = { (float) 3.0, (float) 1.0, (float) 2.0, (float) 5.0 };
		for (float a : x) {
			System.out.print(a + " "); 
		}
		System.out.println();
		NumJava.sub(x, 2);
		for (float a : x) {
			System.out.print(a + " ");
		}
		System.out.println();
		System.out.println("Divide");
		NumJava.divide(matrix1, matrix2).printMe();
		System.out.println("Matrix 1");
		matrix1.printMe();
		System.out.println("Matrix 2");
		matrix2.printMe();
		System.out.println("Sub");
		NumJava.sub(matrix1, matrix2).printMe();
		System.out.println("Divide by val");
		NumJava.divideByVal(matrix1,2).printMe();
		
		NumJava obj2 = new NumJava(3,3);
		obj2.finalmatrix = matrix1.finalmatrix;
		obj2.printMe();
		matrix1.printMe();
		
		System.out.println("Image to column");
		
		float z[] = new float[] { (float)1.0, (float)2.0, (float)1.0, (float)2.0, (float)2.0,
				 (float)0.0, (float)2.0, (float)2.0, (float)1.0, (float)0.0,
				 (float)0.0, (float)2.0, (float)2.0, (float)2.0, (float)0.0,
				 (float)1.0, (float)0.0, (float)0.0, (float)2.0, (float)0.0,
				 (float)1.0, (float)0.0, (float)0.0, (float)1.0, (float)2.0};
		
		float[][] y = new float [1][25];
		y[0] = z;
		
		NumJava newMat = NumJava.pad(y[0], 1, 0, 1, 5, 5);
		newMat.printMe();
		
		NumJava data = new NumJava(9, 9);
//		int odimWidth = 1 + (conv_param.get("imageWidth")-filterW + 2 * pad)/stride; // output dimension of the convolved image.
//		int odimHeight = 1  + (conv_param.get("imageHeight")-filterH + 2 * pad)/stride;

		
		NumJava nw = NumJava.im2col(data, newMat, 3, 3, 1, 3, 1, 0, 2, 1, 5, 5);
		nw.printMe();
		
		System.out.println("Deconvolve");
		NumJava data2 = new NumJava(5, 5);
		NumJava data3 = NumJava.deConvolve(data2, nw, 1, 3, 1, 2, 3, 5, 5);
		data3.printMe();
	}
}
