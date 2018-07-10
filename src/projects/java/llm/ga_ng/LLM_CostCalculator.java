package llm.ga_ng;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ga.CostCalculator;
import llm.LLMNG;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ClusterValidation;

public class LLM_CostCalculator implements CostCalculator<LLM_Individual> {
	
	List<double[]> samples;
	Map<Integer,Set<double[]>> cl;
	int[] fa;
	int ta;
	Dist<double[]> dist;
	
	public LLM_CostCalculator(List<double[]> samples, Map<Integer,Set<double[]>> cl, int[] fa, int ta) {
		this.samples = samples;
		this.cl = cl;
		this.fa = fa;
		this.ta = ta;
		this.dist = new EuclideanDist(fa);
	}

	@Override
	public double getCost(LLM_Individual i) {
		
		double sum = 0;
		for( int j = 0; j < 100; j++ ) {
			LLMNG llmng = i.train(samples, fa, ta,j);
			
			Map<double[],Map<double[],Double>> graph = new HashMap<double[],Map<double[],Double>>();
					
			Map<double[],Set<double[]>> mapping = new HashMap<double[],Set<double[]>>();
			for( double[] n : llmng.getNeurons() )
				mapping.put(n, new HashSet<>() );
			
			for( double[] x : samples ) {
				llmng.present(x);
				List<double[]> neurons = llmng.getNeurons();
				double[] n0 = neurons.get(0);
				double[] n1 = neurons.get(1);
				
				mapping.get(n0).add(x);
				
				if( !graph.containsKey(n0) )
					graph.put( n0, new HashMap<>() );
				if( !graph.containsKey(n1))
					graph.put( n1, new HashMap<>() );
							
				Map<double[],Double> m0 = graph.get( n0 );
				if( !m0.containsKey( n1 ) )
					m0.put( n1, 1.0 );
				else
					m0.put( n1, m0.get(n1) + 1.0 );
				
				Map<double[],Double> m1 = graph.get( n1 );
				if( !m1.containsKey( n0 ) )
					m1.put( n0, 1.0 );
				else
					m1.put( n0, m1.get(n0) + 1.0 );
			}
			
			/*Map<double[],Integer> map = GraphClustering.multilevelOptimization(graph, 10 );
			List<Set<double[]>> ptClusters = new ArrayList<Set<double[]>>( GraphClustering.modulMapToCluster(map) );
			
			// prototype cluster to data cluster
			List<Set<double[]>> clusters = new ArrayList<>();
			for( Set<double[]> ptS : ptClusters ) {
				Set<double[]> s = new HashSet<double[]>();
				for( double[] n0 : ptS )
					s.addAll( mapping.get(n0) );
				clusters.add(s);
			}*/
			Collection<Set<double[]>> clusters = mapping.values();
					
			double nmi = ClusterValidation.getNormalizedMutualInformation( clusters , cl.values() );
			if( Double.isNaN(nmi)) {
				nmi = 0;
				System.err.println("NaN: "+clusters.size());
			}
			sum += 1.0 - nmi;
			
			/*List<Double> response = new ArrayList<>();
			for( double[] x : samples )
				response.add( llmng.present(x)[0] );
			
			int nrSamples = samples.size();
			int nrParams = 0;
			for( double[][] d : llmng.matrix.values() )
				nrParams += d[0].length;
			double mse = SupervisedUtils.getMSE(response, samples, ta);
			
			double aicc = SupervisedUtils.getAICc_GWMODEL(mse, nrParams, nrSamples );
			sum = aicc;*/
		}
		return sum;
	}
}
