package aag_detroit;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

import heuristics.sa.SAIndividual;

public class ClusterIndividual implements SAIndividual<ClusterIndividual> {
	
	Random r = new Random();
	List<Set<double[]>> l;
	
	public ClusterIndividual(List<Set<double[]>> l ) {
		this.l = l;
	}

	@Override
	public void step() {
		Collections.swap(l, r.nextInt(l.size()), r.nextInt(l.size()));
		return;
	}

	@Override
	public ClusterIndividual getCopy() {
		return new ClusterIndividual( new ArrayList<>(l));
	}
	
	public List<Set<double[]>> getList() {
		return l;
	}

}
