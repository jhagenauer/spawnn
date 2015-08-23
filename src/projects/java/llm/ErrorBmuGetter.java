package llm;

import java.util.List;
import java.util.Set;

import spawnn.som.bmu.BmuGetter;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class ErrorBmuGetter extends BmuGetter<double[]> {
	
	private LLMSOM llm;
	private List<double[]> samples, desired;
	
	public ErrorBmuGetter(List<double[]> samples, List<double[]> desired) {
		this.samples = samples;
		this.desired = desired;
	}
	
	@Override
	public GridPos getBmuPos(double[] x, Grid<double[]> grid, Set<GridPos> ign) {
		GridPos bmu = null;
		double minError = Double.MAX_VALUE;
		
		for( GridPos gp : grid.getPositions() ) {
			if( ign.contains( gp) )
				continue;
			
			double[] r = llm.getResponse(x, gp);
			double[] d = desired.get(samples.indexOf(x));
			double error = 0;
			for( int i = 0; i < r.length; i++ )
				error += Math.pow(r[i]-d[i],2);
			if( error < minError || bmu == null ) {
				minError = error;
				bmu = gp;
			}
		}
		return bmu;
	}
	
	public void setLLMSOM( LLMSOM llm ) {
		this.llm = llm;
	}
}
