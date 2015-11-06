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
	}
}
