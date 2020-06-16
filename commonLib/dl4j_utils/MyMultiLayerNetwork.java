/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


package utils.dl4j;

import java.util.ArrayList;
import java.util.Collection;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.heartbeat.Heartbeat;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.heartbeat.utils.EnvironmentUtils;
import org.nd4j.linalg.heartbeat.utils.TaskUtils;

/**
 * MultiLayerNetwork is a neural network with multiple layers in a stack, and
 * usually an output layer. For neural networks with a more complex connection
 * architecture, use {@link org.deeplearning4j.nn.graph.ComputationGraph} which
 * allows for an arbitrary directed acyclic graph connection structure between
 * layers. MultiLayerNetwork is trainable via backprop, with optional
 * pretraining, depending on the type of layers it contains.
 *
 * @author Adam Gibson
 */
public class MyMultiLayerNetwork extends org.deeplearning4j.nn.multilayer.MultiLayerNetwork {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	protected transient MySolver solver; // Used to call optimizers during
											// backprop
	private Collection<TrainingListener> trainingListeners = new ArrayList<>();

	public long trainTime;
	public int currEpoch;

	public MyMultiLayerNetwork(MultiLayerConfiguration conf) {
		super(conf);
		// TODO Auto-generated constructor stub
	}

	public MyMultiLayerNetwork(MultiLayerConfiguration conf, INDArray params) {
		super(conf, params);
	}

	public MyMultiLayerNetwork() {
		super(null);
	}

	/**
	 * Get the updater for this MultiLayerNetwork
	 * 
	 * @return Updater for MultiLayerNetwork
	 */
	public synchronized Updater getUpdater() {
		if (solver == null) {
			solver = new MySolver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
			solver.getOptimizer().setUpdater(UpdaterCreator.getUpdater(this));
		}
		return solver.getOptimizer().getUpdater();
	}

	/** Set the updater for the MultiLayerNetwork */
	public void setUpdater(Updater updater) {
		if (solver == null) {
			solver = new MySolver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
		}
		solver.getOptimizer().setUpdater(updater);
	}

	@Override
	public void fit(DataSetIterator iterator) {
		DataSetIterator iter;
		// we're wrapping all iterators into AsyncDataSetIterator to provide
		// background prefetch - where appropriate
		if (iterator.asyncSupported()) {
			iter = new AsyncDataSetIterator(iterator, 2);
		} else {
			iter = iterator;
		}

		if (trainingListeners.size() > 0) {
			for (TrainingListener tl : trainingListeners) {
				tl.onEpochStart(this);
			}
		}

		if (layerWiseConfigurations.isPretrain()) {
			pretrain(iter);
			if (iter.resetSupported()) {
				iter.reset();
			}
			// while (iter.hasNext()) {
			// DataSet next = iter.next();
			// if (next.getFeatureMatrix() == null || next.getLabels() == null)
			// break;
			// setInput(next.getFeatureMatrix());
			// setLabels(next.getLabels());
			// finetune();
			// }
		}
		if (layerWiseConfigurations.isBackprop()) {
			update(TaskUtils.buildTask(iter));
			if (!iter.hasNext() && iter.resetSupported()) {
				iter.reset();
			}
			while (iter.hasNext()) {
				DataSet next = iter.next();
				if (next.getFeatureMatrix() == null || next.getLabels() == null)
					break;

				boolean hasMaskArrays = next.hasMaskArrays();

				if (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {
					doTruncatedBPTT(next.getFeatureMatrix(), next.getLabels(), next.getFeaturesMaskArray(),
							next.getLabelsMaskArray());
				} else {
					if (hasMaskArrays)
						setLayerMaskArrays(next.getFeaturesMaskArray(), next.getLabelsMaskArray());
					setInput(next.getFeatureMatrix());
					setLabels(next.getLabels());
					if (solver == null) {
						solver = new MySolver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
					}
					solver.optimize();
				}

				if (hasMaskArrays)
					clearLayerMaskArrays();
			}
		} else if (layerWiseConfigurations.isPretrain()) {
			System.err.println("Warning: finetune is not applied.");
		}

		if (trainingListeners.size() > 0) {
			for (TrainingListener tl : trainingListeners) {
				tl.onEpochEnd(this);
			}
		}
	}

	/**
	 * Get the gradients from a miniBatch.
	 * @param miniBatch
	 * @return
	 */
	public Gradient getGradients(DataSet miniBatch) {
				if (miniBatch.getFeatureMatrix() == null || miniBatch.getLabels() == null)
					return null;

				boolean hasMaskArrays = miniBatch.hasMaskArrays();

				if (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {
					doTruncatedBPTT(miniBatch.getFeatureMatrix(), miniBatch.getLabels(), miniBatch.getFeaturesMaskArray(),
							miniBatch.getLabelsMaskArray());
				} else {
					if (hasMaskArrays)
						setLayerMaskArrays(miniBatch.getFeaturesMaskArray(), miniBatch.getLabelsMaskArray());
					setInput(miniBatch.getFeatureMatrix());
					setLabels(miniBatch.getLabels());
					if (solver == null) {
						solver = new MySolver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
					}
					return solver.getGradients();
				}
				return null;
	}

	/**
	 * Update the model by applying a gradient
	 * @param gradient
	 */
	public void applyGradients(Gradient gradient) {
		if (solver == null) {
			solver = new MySolver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
		}
		solver.descent(gradient);
	}

    @Override
    public MyMultiLayerNetwork clone() {
        MultiLayerConfiguration conf = this.layerWiseConfigurations.clone();
        MyMultiLayerNetwork ret = new MyMultiLayerNetwork(conf);
        ret.init(this.params().dup(),false);

        if(solver != null) {
            //If  solver is null: updater hasn't been initialized -> getUpdater call will force initialization, however
            Updater u = this.getUpdater();
            INDArray updaterState = u.getStateViewArray();
            if (updaterState != null) {
                ret.getUpdater().setStateViewArray(ret, updaterState.dup(), false);
            }
        }
        
        ret.currEpoch = this.currEpoch;
        ret.trainTime = this.trainTime;
        
        return ret;
    }
    
	private void update(Task task) {
		if (!initDone) {
			initDone = true;
			Heartbeat heartbeat = Heartbeat.getInstance();
			task = ModelSerializer.taskByModel(this);
			Environment env = EnvironmentUtils.buildEnvironment();
			heartbeat.reportEvent(Event.STANDALONE, env, task);
		}
	}
}
