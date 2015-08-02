package spawnn.ng;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;

public class NG {
	
	protected List<double[]> neurons = null;
	protected Sorter<double[]> sorter;
	protected DecayFunction adaptionRate, stepSizeRate;
	
	public NG( Collection<double[]> neurons, double lInit, double lFinal, double eInit, double eFinal, Sorter<double[]> sorter  ) {
		assert eInit >= 0 && eFinal < eInit;
		
		this.sorter = sorter;				
		this.neurons = new ArrayList<double[]>(neurons);
		this.adaptionRate = new PowerDecay(lInit, lFinal);
		this.stepSizeRate = new PowerDecay(eInit, eFinal);
	}
	
	public NG( Collection<double[]> neurons, DecayFunction adaptionRate, DecayFunction stepSizeRate, Sorter<double[]> sorter  ) {
		this.sorter = sorter;				
		this.neurons = new ArrayList<double[]>(neurons);
	}
	
	@Deprecated
	public NG( int numNeurons, double lInit, double lFinal, double eInit, double eFinal, int dim, Sorter<double[]> bg  ) {
		Random r = new Random();
		
		this.sorter = bg;
		this.adaptionRate = new PowerDecay(lInit, lFinal);
		this.stepSizeRate = new PowerDecay(eInit, eFinal);
				
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
		
	public double[] train( double t, double[] x ) {
		sortNeurons(x);
		
		double l = adaptionRate.getValue(t);
		double e = stepSizeRate.getValue(t);
						
		// adapt
		for( int k = 0; k < neurons.size(); k++ ) {
			double[] w = neurons.get(k);
			double adapt = e * Math.exp( -(double)k/l );
			
			for( int i = 0; i < w.length; i++ ) 
				w[i] +=  adapt * ( x[i] - w[i] ) ;
		} 
		return neurons.get(0);
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
