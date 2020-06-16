/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils;

import java.text.NumberFormat;

import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.geometry.Point;
import org.apache.commons.math3.geometry.Space;
import org.apache.commons.math3.geometry.Vector;

public class ByteVec implements Vector {

	public byte[] v;
	
	private native double getNorm(byte[] vec);
	private native byte[] subtractNative(byte[] a, byte[] b);
	private native byte[] addNative(byte[] a, byte[] b);
	private native byte[] scalarMulNative(byte[] v, double a);

	/**
	 * Flat array byte representation
	 * To be decoded with JNI (first value is the size of the vector)
	 * @param v
	 */
	public ByteVec(byte[] v) {
		this.v = v.clone();
	}
	
	@Override
	public Space getSpace() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isNaN() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double distance(Point p) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public Vector getZero() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double getNorm1() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getNorm() {
		return getNorm(this.v);
	}

	@Override
	public double getNormSq() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getNormInf() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public Vector add(Vector v) {
		ByteVec temp = (ByteVec) v;
		return new ByteVec(addNative(this.v, temp.v));
	}

	@Override
	public Vector add(double factor, Vector v) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Vector subtract(Vector v) {
		ByteVec temp = (ByteVec) v;
		return new ByteVec(subtractNative(this.v, temp.v));
	}
	
	@Override
	public Vector subtract(double factor, Vector v) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Vector negate() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Vector normalize() throws MathArithmeticException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Vector scalarMultiply(double a) {
		return new ByteVec(scalarMulNative(this.v, a));
	}

	@Override
	public boolean isInfinite() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double distance1(Vector v) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double distance(Vector v) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double distanceInf(Vector v) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double distanceSq(Vector v) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double dotProduct(Vector v) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public String toString(NumberFormat format) {
		// TODO Auto-generated method stub
		return null;
	}

}
