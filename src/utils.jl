"""
    Addition of Two  normal random variables assuming
    zero covariance
"""
+(A::MvNormal, B::MvNormal) = MvNormal(a.μ + b.μ, A.Σ + B.Σ)
-(A::MvNormal, B::MvNormal) = MvNormal(a.μ - b.μ, A.Σ + B.Σ)


chol(A::MvNormal) = A.Σ.chol
