class ModifiedArcFaceLoss(nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(ModifiedArcFaceLoss, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin

    def forward(self, query: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor):
        # Normalize input features
        query_norm = query / query.norm(p=2)
        positive_norm = positive / positive.norm(p=2)
        negatives_norm = negatives / negatives.norm(p=2, dim=1, keepdim=True)

        # Compute cosine similarity
        cos_theta_p = torch.dot(query_norm, positive_norm)
        cos_theta_n = torch.matmul(query_norm, negatives_norm.transpose(0, 1))

        # Adjust cosine similarity with margin for positive
        cos_theta_p_m = cos_theta_p * self.cos_m - torch.sqrt(1.0 - cos_theta_p**2) * self.sin_m

        # Select hard negatives: those within the margin from the positive
        hard_negatives = cos_theta_n > cos_theta_p

        # Compute loss for positive
        positive_loss = 1 - cos_theta_p_m

        # Compute loss for hard negatives
        hard_negative_loss = torch.sum(torch.clamp(cos_theta_n[hard_negatives] - cos_theta_p + self.margin, min=0))

        # Combine losses
        loss = positive_loss + hard_negative_loss
        # Scale the loss
        loss = loss * self.s

        return loss
