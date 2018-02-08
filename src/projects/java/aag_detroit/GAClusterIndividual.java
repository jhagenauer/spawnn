package aag_detroit;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import heuristics.ga.GAIndividual;

public class GAClusterIndividual extends ClusterIndividual implements GAIndividual<GAClusterIndividual> {
		
	public GAClusterIndividual(List<Set<double[]>> l ) {
		super(l);
	}
	
	public List<Set<double[]>> getList() {
		return l;
	}

	@Override
	public GAClusterIndividual mutate() {
		List<Set<double[]>> nl = new ArrayList<>(l);
		Collections.swap(nl, r.nextInt(nl.size()), r.nextInt(nl.size()));
		return new GAClusterIndividual(nl);
	}

	@Override
	public GAClusterIndividual recombine(GAClusterIndividual mother) {
		// TODO Auto-generated method stub
		return null;
	}
}
