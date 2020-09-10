/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils.dl4j;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LogSoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import lombok.EqualsAndHashCode;
import lombok.Getter;

/**
 *
 * Multi-Class Cross Entropy loss function:<br>
 * L = sum_i actual_i * log( predicted_i )
 *
 * @see LossNegativeLogLikelihood
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter
public class CustomCrossEntropy implements ILossFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	@JsonSerialize(using = RowVectorSerializer.class)
	@JsonDeserialize(using = RowVectorDeserializer.class)
	private final INDArray weights;

	public CustomCrossEntropy() {
		this(null);
	}

	/**
	 * Multi-Class Cross Entropy loss function where each the output is
	 * (optionally) weighted/scaled by a fixed scalar value. Note that the
	 * weights array must be a row vector, of length equal to the labels/output
	 * dimension 1 size. A weight vector of 1s should give identical results to
	 * no weight vector.
	 *
	 * @param weights
	 *            Weights array (row vector). May be null.
	 */
	public CustomCrossEntropy(INDArray weights) {
		if (weights != null && !weights.isRowVector()) {
			throw new IllegalArgumentException("Weights array must be a row vector");
		}
		this.weights = weights;
	}

	private static INDArray scoreArray(INDArray labels, INDArray output) {
		INDArray scoreArr;

		// clipping
		double upper = 1.0;
		double lower = 1e-20;
		BooleanIndexing.replaceWhere(output, upper, Conditions.greaterThan(upper));
		BooleanIndexing.replaceWhere(output, lower, Conditions.lessThan(lower));

		scoreArr = Transforms.log(output, false).muli(labels);

		return scoreArr;
	}

	private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		INDArray scoreArr;
		// if ("softmax".equals(activationFn)) {
		if (activationFn instanceof ActivationSoftmax) {
			// Use LogSoftMax op to avoid numerical issues when calculating
			// score
			INDArray logsoftmax = Nd4j.getExecutioner().execAndReturn(new LogSoftMax(preOutput.dup()));
			scoreArr = logsoftmax.muli(labels);

		} else {
			// INDArray output =
			// Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn,
			// preOutput.dup()));
			INDArray output = activationFn.getActivation(preOutput.dup(), true);

			// clipping
			double upper = 1.0;
			double lower = 1e-20;
			BooleanIndexing.replaceWhere(output, upper, Conditions.greaterThan(upper));
			BooleanIndexing.replaceWhere(output, lower, Conditions.lessThan(lower));

			scoreArr = Transforms.log(output, false).muli(labels);
		}

		// Weighted loss function
		if (weights != null) {
			if (weights.length() != scoreArr.size(1)) {
				throw new IllegalStateException("Weights vector (length " + weights.length()
						+ ") does not match output.size(1)=" + preOutput.size(1));
			}
			scoreArr.muliRowVector(weights);
		}

		if (mask != null) {
			scoreArr.muliColumnVector(mask);
		}
		return scoreArr;
	}

	@Override
	public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
			boolean average) {
		INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

		double score = -scoreArr.sumNumber().doubleValue();

		if (average) {
			score /= scoreArr.size(0);
		}

		return score;
	}

	public static double computeScore(INDArray labels, INDArray output, boolean average) {
		INDArray scoreArr = scoreArray(labels, output);

		double score = -scoreArr.sumNumber().doubleValue();

		if (average) {
			score /= scoreArr.size(0);
		}

		return score;
	}

	@Override
	public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
		return scoreArr.sum(1).muli(-1);
	}

	@Override
	public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		INDArray grad;
		// INDArray output =
		// Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn,
		// preOutput.dup()));
		INDArray output = activationFn.getActivation(preOutput.dup(), true);

		if (activationFn instanceof ActivationSoftmax) {
			// Weighted loss function
			if (weights != null) {
				if (weights.length() != output.size(1)) {
					throw new IllegalStateException("Weights vector (length " + weights.length()
							+ ") does not match output.size(1)=" + output.size(1));
				}
				INDArray temp = labels.mulRowVector(weights);
				INDArray col = temp.sum(1);
				grad = output.mulColumnVector(col).sub(temp);
			} else {
				grad = output.subi(labels);
			}
		} else {
			// INDArray sigmaPrimeZ =
			// Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn,
			// preOutput.dup()).derivative());

			INDArray dLda = output.rdivi(labels).negi();
			grad = activationFn.backprop(preOutput, dLda).getKey(); // TODO
																		// activation
																		// function
																		// with
																		// weights

			// Weighted loss function
			if (weights != null) {
				if (weights.length() != output.size(1)) {
					throw new IllegalStateException("Weights vector (length " + weights.length()
							+ ") does not match output.size(1)=" + output.size(1));
				}
				grad.muliRowVector(weights);
			}
		}

		// Loss function with masking
		if (mask != null) {
			grad.muliColumnVector(mask);
		}

		return grad;
	}

	@Override
	public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
			INDArray mask, boolean average) {
		// TODO: probably a more efficient way to do this...

		return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
				computeGradient(labels, preOutput, activationFn, mask));
	}

	@Override
	public String toString() {
		if (weights == null)
			return "LossMCXENT()";
		return "LossMCXENT(weights=" + weights + ")";
	}

	
}
