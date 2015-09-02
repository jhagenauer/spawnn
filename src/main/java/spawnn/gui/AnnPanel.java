package spawnn.gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.miginfocom.swing.MigLayout;

public class AnnPanel extends JPanel implements ChangeListener, ActionListener {
	
	private static final long serialVersionUID = -5917163707063857622L;
	public static final String TRAIN_PROP = "train", APPLY_EXISTING = "apply_existing";
	
	protected JTabbedPane tpANN, tpContextModel;
	protected JTextField trainingCycles, runs;
	protected JPanel somPanel, ngPanel;
	protected JPanel nonePanel, weightedPanel, geoSomPanel, cngPanel, wmcPanel, augmentedPanel;
	private JButton btnTrain, btnApply;
	private boolean contextModelsEnabled = false;
		
	private static final int NONE = 0, AUGMENTED = 1, WEIGHTED = 2, GEOSOM = 3, CNG = 4, WMC = 5;
	
	public AnnPanel() {
		super();
		setLayout( new MigLayout("") );
		
		tpANN = new JTabbedPane();
		tpANN.setBorder( BorderFactory.createTitledBorder("Self-Organizing Neural Network") );

		somPanel = new SOMPanel();
		tpANN.addTab("SOM", somPanel );
		
		ngPanel = new NGPanel();
		tpANN.addTab("NG", ngPanel );
		
		tpANN.addChangeListener( this );
				
		// None, Weighted, GeoSOM, CNG, MDMNG
		tpContextModel = new JTabbedPane();
		tpContextModel.setBorder( BorderFactory.createTitledBorder("Spatial Context Model") );
		
		nonePanel = new JPanel();
		tpContextModel.addTab("None", nonePanel);
		
		augmentedPanel = new AugmentedPanel();
		tpContextModel.addTab("Augmented", augmentedPanel);
		
		weightedPanel = new WeightedPanel();
		tpContextModel.addTab("Weighted", weightedPanel );
		
		geoSomPanel = new GeoSOMPanel();
		tpContextModel.addTab("GeoSOM", geoSomPanel );
		
		cngPanel = new CNGPanel();
		tpContextModel.addTab("CNG", cngPanel );
		
		wmcPanel = new WMCPanel();
		tpContextModel.addTab("WMC",wmcPanel );
						
		trainingCycles = new JTextField();
		trainingCycles.setText("100000");
		trainingCycles.setColumns(10);
				
		runs = new JTextField();
		runs.setText("1");
		runs.setColumns(2);
						
		btnTrain = new JButton("TRAIN!");
		btnTrain.addActionListener(this);
		
		add( tpANN, "w 50%, span 2, grow, wrap" );
		add(tpContextModel,"span 2, grow, wrap");
		add( new JLabel("Training cycles: "), "split 4" );
		add( trainingCycles, "" );
		add( new JLabel("Runs:"));
		add( runs, "" );
		
		//TODO comment me out plz!!!
		/*btnApply = new JButton("Apply existing...");
		btnApply.addActionListener(this);
		add( btnApply,"split 2, align right");*/
		
		add( btnTrain, "align right" );	
	}

	@Override
	public void stateChanged(ChangeEvent e) {
		if( e.getSource() == tpANN ) {
			if( tpANN.getSelectedComponent() == somPanel ) {
				updateContextModelsEnabled();
			} else if( tpANN.getSelectedComponent() == ngPanel ) {
				if( tpContextModel.getSelectedComponent() == geoSomPanel )
					tpContextModel.setSelectedComponent(nonePanel);
				
				updateContextModelsEnabled();
			}
		}
	}
	
	private void updateContextModelsEnabled() {
		tpContextModel.setEnabledAt(AUGMENTED, contextModelsEnabled);
		tpContextModel.setEnabledAt(WEIGHTED, contextModelsEnabled);
		if( tpANN.getSelectedComponent() == somPanel )
			tpContextModel.setEnabledAt(GEOSOM, contextModelsEnabled);
		else
			tpContextModel.setEnabledAt(GEOSOM, false);
		tpContextModel.setEnabledAt(CNG,contextModelsEnabled);	
	}
	
	public void enableCentroidBasedContextModels(boolean b) {
		contextModelsEnabled = b;
		updateContextModelsEnabled();
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		if( e.getSource() == btnTrain ) 
			firePropertyChange(TRAIN_PROP, false, true);
		else if( e.getSource() == btnApply ) {
			firePropertyChange(APPLY_EXISTING, false, true);
		}
	}
	
	public int getNumTraining() {
		return Integer.parseInt(trainingCycles.getText() );
	}
	
	public int getRuns() {
		return Integer.parseInt(runs.getText());
	}
}
