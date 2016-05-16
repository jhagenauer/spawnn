package regionalization.medoid.ga;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import heuristics.GAIndividual;
import regionalization.medoid.MedoidRegioClustering;
import regionalization.medoid.MedoidRegioClustering.GrowMode;
import spawnn.dist.Dist;

public class MedoidIndividual extends GAIndividual<MedoidIndividual> {
	
	private List<Integer> c;
	int numSamples;
	Random r = new Random();
	
	public MedoidIndividual(List<Integer> c, int numSamples) {
		this.c = c;
		this.numSamples = numSamples;
	}

	public MedoidIndividual(int numCluster, int numSamples) {
		c = new ArrayList<Integer>();
		while( c.size() != numCluster ) {
			int i = r.nextInt(numSamples);
			if( !c.contains(i) )
				c.add(i);
		}
		this.numSamples = numSamples;
	}

	@Override
	public void mutate() {
		int i = r.nextInt(numSamples);
		if( !c.contains(i) ) {
			c.remove(r.nextInt(c.size()));
			c.add(i);
		}
	}

	@Override
	public MedoidIndividual recombine(MedoidIndividual mother) {
		List<Integer> n = new ArrayList<Integer>(c);
		for( int i : mother.getChromosome() ) {
			if( !n.contains(i) ) 
				n.add(i);
		}
		while( n.size() > c.size() ) 
			n.remove(r.nextInt(n.size()));
		return new MedoidIndividual(c, numSamples);
	}
	
	public List<Integer> getChromosome() {
		return c;
	}
	
	public Collection<Set<double[]>> toCluster( List<double[]> samples, Map<double[],Set<double[]>> cm, Dist<double[]> dist, GrowMode dm ) {
		Set<double[]> medoids = new HashSet<double[]>();
		for( int i : c )
			medoids.add(samples.get(i));
		return MedoidRegioClustering.growFromMedoids(cm, medoids, dist, dm).values();
	}
}
