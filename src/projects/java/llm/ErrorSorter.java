package llm;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import spawnn.ng.sorter.Sorter;

public class ErrorSorter implements Sorter<double[]> {
	
	private LLMNG llm;
	private List<double[]> samples, desired;
	
	public ErrorSorter(List<double[]> samples, List<double[]> desired) {
		this.samples = samples;
		this.desired = desired;
	}
	
	@Override
	public void sort( final double[] x, List<double[]> neurons ) {
		Collections.sort(neurons, new Comparator<double[]>() {
			@Override
			public int compare(double[] o1, double[] o2) {
				double[] d = desired.get(samples.indexOf(x));
				double[] r1 = llm.getResponse(x, o1);
				double[] r2 = llm.getResponse(x, o2);
				
				double e1 = 0;
				for( int i = 0; i < r1.length; i++ )
					e1 += Math.pow(r1[i]-d[i],2);
				
				double e2 = 0;
				for( int i = 0; i < r2.length; i++ )
					e2 += Math.pow(r2[i]-d[i],2);
								
				if( e1 < e2 ) return -1;				
				else if( e1 > e2 ) return 1;
				else return 0; }
		});
	}
	
	public void setLLMNG( LLMNG llm ) {
		this.llm = llm;
	}
}
