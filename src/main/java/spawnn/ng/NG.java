package spawnn.ng;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import spawnn.UnsupervisedNet;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;

public class NG implements UnsupervisedNet {
	
	protected List<double[]> neurons = null;
	protected Sorter<double[]> sorter;
	protected DecayFunction neighborhoodRange, adaptationRate;
	
	public NG( List<double[]> neurons, double lInit, double lFinal, double eInit, double eFinal, Sorter<double[]> sorter  ) {
		this( neurons,new PowerDecay(lInit, lFinal), new PowerDecay(eInit, eFinal),sorter);
		assert eInit >= 0 && eFinal < eInit;
	}
	
	public NG( List<double[]> neurons, DecayFunction neighborhoodRange, DecayFunction adaptationRate, Sorter<double[]> sorter  ) {
		this.sorter = sorter;				
		this.neurons = new ArrayList<double[]>(neurons);
		this.neighborhoodRange = neighborhoodRange;
		this.adaptationRate = adaptationRate;
	}
	
	@Deprecated
	public NG( int numNeurons, double lInit, double lFinal, double eInit, double eFinal, int dim, Sorter<double[]> sorter  ) {
		Random r = new Random();
		
		this.sorter = sorter;
		this.neighborhoodRange = new PowerDecay(lInit, lFinal);
		this.adaptationRate = new PowerDecay(eInit, eFinal);
				
		this.neurons = new ArrayList<double[]>();
				
		for( int i = 0; i < numNeurons; i++ ) {
			double[] d = new double[dim];
			
			// init randomly
			for( int j = 0; j < dim; j++ )
				d[j] = r.nextDouble();
			neurons.add( d  );
		}
	}
	
	public void sortNeurons(double[] x ) {
		sorter.sort(x, neurons);
	}
		
	public void train( double t, double[] x ) {
		sortNeurons(x);
		
		double l = neighborhoodRange.getValue(t);
		double e = adaptationRate.getValue(t);
						
		// adapt
		for( int k = 0; k < neurons.size(); k++ ) {
			double[] w = neurons.get(k);
			double adapt = e * Math.exp( -(double)k/l );
						
			for( int i = 0; i < w.length; i++ ) 
				w[i] +=  adapt * ( x[i] - w[i] ) ;
		} 
		return;
	}
		
	public List<double[]> getNeurons() {
		return neurons;
	}
	
	public void initRandom( List<double[]> samples ) {
		Random r = new Random();
		for(double[] n : neurons )  
			n = Arrays.copyOf(samples.get(r.nextInt(samples.size() ) ), n.length );
	}
}
