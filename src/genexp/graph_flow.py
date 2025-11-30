from maxentdiff.models import FlowModel, InterpolantScheduler, AdjointState
from maxentdiff.sampling import Sampler
from flowmol import FlowMol
from flowmol.models.interpolant_scheduler import InterpolantScheduler as FlowMolInterpolantScheduler


class GraphInterpolantScheduler(InterpolantScheduler):
    def __init__(self, scheduler: FlowMolInterpolantScheduler):
        super().__init__()
        self.scheduler = scheduler
    
    def beta_t(self, t):
        return self.scheduler.beta_t(t)
    
    def beta_t_prime(self, t):
        return self.scheduler.beta_t_prime(t)
    
    def alpha_t(self, t):
        return self.scheduler.alpha_t(t)
    
    def alpha_t_prime(self, t):
        return self.scheduler.alpha_t_prime(t)
    
    def interpolants(self, t):
        return self.alpha_t(t), self.beta_t(t)
    
    def interpolants_prime(self, t):
        return self.alpha_t_prime(t), self.beta_t_prime(t)
    
    def eta_t(self, t):
        at, bt = self.interpolants(t)
        atp, btp = self.interpolants_prime(t)
        kt = atp / at
        return bt * (kt * bt - btp)

    def memoryless_sigma_t(self, t):
        return torch.sqrt(2 * self.eta_t(t))


class GraphFlowModel(FlowModel):
    """
    Wrapper for FlowMol
    """
    def __init__(self, model: FlowMol):
        super().__init__(model, GraphInterpolantScheduler(model.interpolant_scheduler))
    

    def velocity_field(self, g, t, ue_mask=None):
        node_batch_idx = torch.zeros(g_t.num_nodes(), dtype=torch.long)
        upper_edge_mask = g_t.edata['ue_mask'] if ue_mask is None else ue_mask

        dst_dict = model.vector_field(
            g_t, 
            t=torch.full((g_t.batch_size,), t, device=g_t.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g_t.ndata['x_t']

        v_pred = model.vector_field.vector_field(x_t, x_1, alpha, alpha_dot)
        return v_pred


class GraphEulerMaruyamaSampler(Sampler):
    def __init__(self, model: GraphFlowModel, sampler_type=None):
        super().__init__(model)
        self.sampler_type = sampler_type
    

    def sample_init_dist(self, N=1, device=None):
        return self.model.model.sample_n_atoms(N)


    def sample_trajectories(self, N=1, T=1000, n_atoms=None, sampler_type=None):
        """
        Sample N trajectories of length T using memoryless sampling
        """
        atoms_per_mol = self.sample_init_dist(N)
        if n_atoms is not None:
            atoms_per_mol = atoms_per_mol * 0 + n_atoms

        _, graph_trajectories = self.model.model.sample(atoms_per_mol,
                n_timesteps = T,     
                sampler_type = self.sampler_type if sampler_type is None else sampler_type,
                device = self.model.device,
                keep_intermediate_graphs = True,
            )

        return graph_trajectories